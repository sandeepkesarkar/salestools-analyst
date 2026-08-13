import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from marimo_ask import query_ollama, run_generated_code
    return mo, query_ollama, run_generated_code


@app.cell
def __(mo):
    mo.md(
        """
        # salestools-analyst — Marimo `%%ask` equivalent

        Marimo has no cell-magic system, so this is two explicit steps instead of one
        `%%ask` cell: **Ask** generates the code and shows it to you (editable) before
        anything runs; **Run this code** is a separate action that actually executes it.
        """
    )
    return


@app.cell
def __(mo):
    question = mo.ui.text(
        label="Question",
        placeholder="Is my overall sales trend going up or down?",
        full_width=True,
    )
    model = mo.ui.dropdown(
        options=["sales-analyst-1.5b", "sales-analyst-3b", "sales-analyst-1.5b-v2"],
        value="sales-analyst-1.5b",
        label="Model",
    )
    csv_path = mo.ui.text(
        label="CSV path",
        value="tests/fixtures/multi_product.csv",
        full_width=True,
    )
    ask_button = mo.ui.run_button(label="Ask")
    mo.vstack([question, model, csv_path, ask_button])
    return ask_button, csv_path, model, question


@app.cell
def __(ask_button, mo, model, question, query_ollama):
    mo.stop(not ask_button.value, mo.md("Click **Ask** to generate code."))
    mo.stop(not question.value.strip(), mo.md("Type a question first."))

    code, error = query_ollama(question.value, model=model.value)

    code_editor = mo.ui.code_editor(value=code or f"# {error}", language="python")
    run_button = mo.ui.run_button(label="Run this code")

    mo.vstack(
        [
            mo.md(f"**Error:** {error}") if error else mo.md("Generated code (review, edit if you like):"),
            code_editor,
            run_button,
        ]
    )
    return code_editor, run_button


@app.cell
def __(code_editor, csv_path, mo, run_button, run_generated_code):
    mo.stop(not run_button.value, mo.md("Click **Run this code** to execute it."))

    result = run_generated_code(code_editor.value, csv_path.value)

    output = []
    if result["stdout"]:
        output.append(mo.md(f"```\n{result['stdout']}\n```"))
    if result["error"]:
        output.append(mo.md(f"**Execution error:** {result['error']}"))
    if result["fig"] is not None:
        output.append(result["fig"])

    mo.vstack(output) if output else mo.md("*(no output — this is expected for a scope-refusal answer, which has nothing to run)*")
    return


if __name__ == "__main__":
    app.run()
