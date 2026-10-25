from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.panel import Panel

console = Console()


class DebugPromptHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        for conversation in messages:
            console.print(
                Panel.fit(
                    "\n".join(
                        [f"[bold]{m.type}[/bold]: {m.content}" for m in conversation]
                    ),
                    title="[yellow]DEBUG: Prompt[/yellow]",
                    border_style="yellow",
                )
            )
