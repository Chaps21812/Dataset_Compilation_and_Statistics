import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _(TOTAL_UI):
    TOTAL_UI
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    # Cell 1: Create the tab selector
    tabs = mo.ui.tabs({
        "Data Viewer": "📊 Data Viewer",
        "Download Data": "⬇️ Downloader",
        "Error Characterization": "❌ Error Characterization",
        "COCO Tools": "💽 COCO Converter",
        "Utilities": "⚙️ Utilities"
    })



    return mo, tabs


@app.cell
def _(clear_cache_plz, mo, recalculate_statistics_plz, tabs, view_folders_plz):

    if tabs.value == "Data Viewer":
        title = mo.md("Please enter your directory")
        file_directory = mo.ui.text(placeholder="/path/to/folder")

        view = mo.vstack([title, file_directory,
            mo.md(f"Enter some text: {file_directory}")
        ])

    elif tabs.value == "Download Data":
        main_page = mo.vstack([
            mo.md("# View AWS Directory"),
            mo.md("# Select Download Directories"),
            mo.md("# View AWS Directory"),
            mo.md("# View AWS Directory"),
        ])

        view = mo.vstack([
            mo.ui.text("## 📊 Data Controls"),
            mo.hstack([
                mo.ui.slider(0, 100, value=25, label="Threshold"),
                mo.ui.text("Label Filter"), 
                main_page
            ])
        ])

    elif tabs.value == "Error Characterization":
        view = mo.vstack([
            mo.ui.text("## ⚙️ Settings Panel"),
            mo.hstack([
                mo.ui.checkbox("Enable feature A"),
                mo.ui.checkbox("Enable feature B")
            ])
        ])

    elif tabs.value == "COCO Tools":
        view = mo.vstack([
            mo.ui.text("## 📁 File Controls"),
            mo.hstack([
                mo.ui.button("Upload"),
                mo.ui.button("Download")
            ])
        ])

    elif tabs.value == "Utilities":
        clear_cache_button = mo.ui.button(on_click=clear_cache_plz)
        view_directory_button = mo.ui.button(on_click=view_folders_plz)
        recalculate_statistics_button = mo.ui.button(on_click=recalculate_statistics_plz)

        view = mo.vstack([
            mo.md("# View Directory"),view_directory_button,
            mo.md("# Clear Cache"),clear_cache_button,
            mo.md("# Recalculate All Statistics"),recalculate_statistics_button
        ])

    TOTAL_UI = mo.vstack([tabs, view])
    return (TOTAL_UI,)


@app.cell
def _():
    from utilities import get_folders_in_directory, clear_cache, view_folders, recalculate_statistics
    from data_download_tool import download_data, summarize_s3_structure
    from annotation_viewer import plot_annotations, plot_annotation_subset
    from pandas_statistics import file_path_loader

    from aws_s3_viewer import S3Client
    import os

    import io
    import contextlib

    def my_function():
        with contextlib.redirect_stdout(io.StringIO()) as f:
            print("Hello from inside the function!")
            print("More output here...")
        return f.getvalue()

    GLOBAL_FOLDER = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset_Statistics/data"
    def clear_cache_plz(bruh):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            clear_cache(GLOBAL_FOLDER)
        return f.getvalue()
    def view_folders_plz(bruh):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            view_folders(GLOBAL_FOLDER)
        return f.getvalue()
    def recalculate_statistics_plz(bruh):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            recalculate_statistics(GLOBAL_FOLDER)
        return f.getvalue()
    return clear_cache_plz, recalculate_statistics_plz, view_folders_plz


if __name__ == "__main__":
    app.run()
