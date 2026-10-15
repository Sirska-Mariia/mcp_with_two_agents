import json
import time
from dataclasses import dataclass, field
from datetime import date

from llm import LLMError, assistant_message
from mcp_client import McpHandle, Tracer

PREPARER_TOOLS = {"get_policy", "list_expenses"}          
VERIFIER_TOOLS = {"get_policy", "submit_expense"}    
SINGLE_TOOLS = {"get_policy", "list_expenses", "submit_expense"}

COMMON = """Сьогодні {today}. Поточний користувач: {employee}.
Валюта завжди гривня. Відносні дати переводь у YYYY-MM-DD, виходячи з сьогоднішньої дати.
Категорії: meals, transport, office, training. Не вигадуй категорій і правил: бери їх з інструментів."""

PREPARER_PROMPT = """Ти Підготовлювач заявок на витрати. Ти НЕ можеш нічого записувати.
Твоя робота: розібрати запит користувача в чернетки.
1. Для кожної витрати визнач category, amount_uah, date (YYYY-MM-DD), description, has_receipt.
2. Викличи get_policy для категорії, а list_expenses (один раз) для пошуку дубліката.
   Роби якомога менше викликів: по одному get_policy на категорію.
3. Дай вердикт на кожну чернетку: "ok" або "reject" з причиною.
Якщо категорія в запиті не збігається з допустимими, підбери найближчу за змістом і напиши це в reason.
Відповідай ТІЛЬКИ JSON-масивом об'єктів, без markdown:
[{{"category": "...", "amount_uah": 0, "date": "YYYY-MM-DD", "description": "...",
  "has_receipt": true, "verdict": "ok|reject", "reason": "..."}}]
""" + COMMON

VERIFIER_PROMPT = """Ти Перевіряльник. Ти ЄДИНИЙ, хто може записувати витрати (submit_expense).
Тобі дають оригінальний запит і чернетки Підготовлювача. Не довіряй чернетці наосліп:
1. Звір кожну чернетку з оригінальним запитом (сума, дата, категорія, чек).
2. Звір з політикою через get_policy.
3. Тільки якщо все сходиться, виклич submit_expense. Якщо вердикт reject або правила порушені, НЕ записуй.
4. Якщо submit_expense повернув помилку з retryable=true і hint підказує виправлення, виправ і спробуй ще раз
   (не більше одного повтору). Якщо retryable=false, не повторюй: поясни користувачу причину.
Фінальна відповідь користувачу: коротко, українською, по кожній витраті що записано або чому відмовлено.
""" + COMMON

SINGLE_PROMPT = """Ти агент подання витрат. Розбери запит користувача, перевір політику (get_policy),
перевір дублікати (list_expenses) і подай витрату (submit_expense), якщо правила дозволяють.
Якщо submit_expense повернув помилку з retryable=true і hint підказує виправлення, виправ і повтори
(не більше одного разу). Якщо retryable=false, не повторюй і поясни причину.
Фінальна відповідь: коротко, українською, по кожній витраті що записано або чому відмовлено.
""" + COMMON


@dataclass
class AgentResult:
    text: str = ""
    tool_calls: int = 0
    steps: int = 0
    seconds: float = 0.0
    halted: bool = False
    error: str = ""


@dataclass
class RunResult:
    answer: str
    tool_calls: int = 0
    model_calls: int = 0   
    seconds: float = 0.0
    parts: list[AgentResult] = field(default_factory=list)


async def run_agent(client, name: str, system: str, user_text: str, mcp: McpHandle,
                    tool_defs: list[dict], allowed: set[str], tracer: Tracer,
                    trace_id: str, max_steps: int = 12) -> AgentResult:
    """Цикл агента: decide -> act (через MCP) -> observe, з бюджетом кроків."""
    tools = [t for t in tool_defs if t["function"]["name"] in allowed]
    msgs = [{"role": "user", "content": user_text}]
    out = AgentResult()
    t0 = time.time()
    for step in range(max_steps):
        out.steps = step + 1
        try:
            reply = await client.chat(system, msgs, tools)
        except LLMError as e:     
            out.halted, out.error = True, str(e)
            out.text = f"Помилка моделі: {e}"
            out.seconds = time.time() - t0
            return out
        if not reply.tool_calls:
            out.text = reply.content or ""
            out.seconds = time.time() - t0
            return out
        msgs.append(assistant_message(reply))
        for tc in reply.tool_calls:
            t1 = time.time()
            try:
                args = json.loads(tc.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments не є об'єктом")
            except (json.JSONDecodeError, ValueError):
                args = {"_raw": (tc.arguments or "")[:200]}
                res = {"error": "bad_arguments", "message": "Аргументи не є валідним JSON-об'єктом.",
                       "hint": "Передай аргументи як JSON за схемою інструмента", "retryable": True}
            else:
                if tc.name not in allowed: 
                    res = {"error": "permission_denied",
                           "message": f"Агент {name} не має права викликати {tc.name}.",
                           "hint": "Передай задачу агенту з відповідними правами",
                           "retryable": False}
                else:
                    res = await mcp.call(tc.name, args)
            out.tool_calls += 1
            tracer.log(trace_id=trace_id, agent=name, tool=tc.name, args=args,
                       latency_ms=int((time.time() - t1) * 1000), result=res,
                       ok="error" not in res)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(res, ensure_ascii=False)})
    out.halted = True
    out.text = "Зупинено: вичерпано ліміт кроків."
    out.seconds = time.time() - t0
    return out


def _fmt(prompt: str, employee: str) -> str:
    return prompt.format(today=date.today().isoformat(), employee=employee)


async def run_two_agents(client, mcp: McpHandle, query: str, employee: str,
                         tracer: Tracer, trace_id: str) -> RunResult:
    defs = await mcp.tool_defs()
    a = await run_agent(client, "preparer", _fmt(PREPARER_PROMPT, employee), query, mcp,
                        defs, PREPARER_TOOLS, tracer, trace_id)
    if a.error: 
        return RunResult(a.text, a.tool_calls, a.steps, a.seconds, [a])
    handoff = f"Оригінальний запит користувача:\n{query}\n\nЧернетки Підготовлювача:\n{a.text}"
    b = await run_agent(client, "verifier", _fmt(VERIFIER_PROMPT, employee), handoff, mcp,
                        defs, VERIFIER_TOOLS, tracer, trace_id)
    return RunResult(b.text, a.tool_calls + b.tool_calls, a.steps + b.steps,
                     a.seconds + b.seconds, [a, b])


async def run_single_agent(client, mcp: McpHandle, query: str, employee: str,
                           tracer: Tracer, trace_id: str) -> RunResult:
    defs = await mcp.tool_defs()
    r = await run_agent(client, "single", _fmt(SINGLE_PROMPT, employee), query, mcp,
                        defs, SINGLE_TOOLS, tracer, trace_id)
    return RunResult(r.text, r.tool_calls, r.steps, r.seconds, [r])
