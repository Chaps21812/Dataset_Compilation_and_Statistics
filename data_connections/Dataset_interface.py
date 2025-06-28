import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    from data_connections import s3viewer
    return


@app.cell
def _():
    return

@app.cell
def _():
    checkbox = app.ui.checkbox("Enable feature")
    return 



if __name__ == "__main__":
    app.run()
