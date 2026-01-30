"""
Tkinter-based GUI for loading, analyzing, training, and visualizing socio-economic data.  
Includes data inspection, model training with Random Forest, and various statistical plots.  
User-configurable via a separate INI file for theming and behavior.
"""

import io
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import configparser
import pandas as pd
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from Library.data_utils import load_and_clean_data, inspect_data, split_features_target
from Scripts.model_utils import (
    build_model_pipeline, train_model, evaluate_model,
    save_model, save_report)
from Library.plot_utils import (
    plot_age_distribution, plot_hours_per_week_distribution,
    plot_education_level_pie, plot_workclass_distribution,
    plot_income_distribution, plot_age_vs_income,
    plot_marital_status_distribution, plot_occupation_treemap,
    plot_race_pie, plot_gender_distribution,
    plot_education_vs_income, plot_occupation_vs_income_heatmap,
    plot_correlation_heatmap)

def resource_path(relative_path: str) -> str:
    """
    Absolute path to bundled resource (PyInstaller) or project resource (dev).
    relative_path should use forward slashes or os.path.join parts like: "Data/adult.csv"
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        # project root (one level up from Scripts/)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_path, relative_path)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = resource_path(os.path.join("Data", "adult.csv"))

def writable_config_path() -> str:
    """
    Store config.ini in a user-writable location when running as EXE.
    """
    if getattr(sys, "frozen", False):
        app_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "IncomePredictor")
        os.makedirs(app_dir, exist_ok=True)
        return os.path.join(app_dir, "config.ini")
    # dev mode: use Scripts/config.ini
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "Scripts", "config.ini")

CONFIG_PATH = writable_config_path()


FONT_FAMILY = FONT_SIZE = SIDEBAR_FONT_SIZE = None
REPORT_BUTTON_COLOR = BUTTON_COLOR = None
BG_COLOR = None

def load_config():
    """Load the interface configuration from the INI file."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    if 'interface' not in config:
        config['interface'] = {}
    return config

def save_config(config):
    """Save the interface configuration to the INI file."""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

def apply_config():
    """Apply values from config.ini to global UI settings."""
    global FONT_FAMILY, FONT_SIZE, SIDEBAR_FONT_SIZE
    global REPORT_BUTTON_COLOR, BUTTON_COLOR
    global BG_COLOR

    config = load_config()
    interface = config['interface']

    def safe_get(key, default):
        val = interface.get(key, default)
        return val if val else default

    FONT_FAMILY = safe_get("font_family", "Arial")
    FONT_SIZE = safe_get("font_size", "14")
    SIDEBAR_FONT_SIZE = safe_get("sidebar_font_size", "14")
    REPORT_BUTTON_COLOR = safe_get("report_button_color", "yellow")
    BUTTON_COLOR = safe_get("button_color", "blue")
    BG_COLOR = safe_get("bg_color", "white")

def _graphics_output_dir():
    """Return the absolute path to the graphics output directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Work', 'Graphics'
    )

def data_page(page_frame):
    """Creates the Data tab with Load & Clean Data functionality only."""
    # === Frame setup ===
    data_page_frame = tk.Frame(page_frame, bg=BG_COLOR)
    data_page_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)

    data_page_frame.columnconfigure(0, weight=1)
    data_page_frame.rowconfigure(1, weight=0)
    data_page_frame.rowconfigure(2, weight=1)

    # === Top buttons ===
    button_frame = tk.Frame(data_page_frame, bg=BG_COLOR)
    button_frame.grid(row=0, column=0, pady=10, sticky="ew")
    button_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)

    # === Status bar ===
    status_label = tk.Label(
        data_page_frame, text="", bg=BG_COLOR, fg="green",
        font=(FONT_FAMILY, FONT_SIZE, "bold")
    )
    status_label.grid(row=1, column=0, sticky="ew", pady=5)

    # === Table display area ===
    content_frame = tk.Frame(data_page_frame, bg=BG_COLOR)
    content_frame.grid(row=2, column=0, sticky="nsew")
    content_frame.columnconfigure(0, weight=1)
    content_frame.rowconfigure(0, weight=1)

    # === Utilities ===
    def clear_content():
        for widget in content_frame.winfo_children():
            widget.destroy()

    def update_status(text, color="green"):
        status_label.config(text=text, fg=color)
        status_label.update_idletasks()

    def display_table(df):
        clear_content()
        if df.empty:
            return

        x_scroll = tk.Scrollbar(content_frame, orient="horizontal")
        y_scroll = tk.Scrollbar(content_frame, orient="vertical")
        tree = ttk.Treeview(
            content_frame,
            columns=list(df.columns),
            show="headings",
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set
        )

        x_scroll.config(command=tree.xview)
        y_scroll.config(command=tree.yview)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")

        for _, row in df.iterrows():
            tree.insert("", tk.END, values=list(row))

        data_page.tree = tree

    # === Button action ===
    def handle_load_data():
        update_status("🔄 Loading and cleaning data...", "orange")
        try:
            df = load_and_clean_data(CSV_PATH)
            data_page.df = df  # Save for later use
            display_table(df)
            update_status(f"✅ Loaded {len(df)} rows.", "green")
            inspect_btn.grid(row=0, column=1, sticky="ew", padx=5)
            inspect_btn.lift()
        except Exception as e:
            update_status(f"❌ Failed to load data: {e}", "red")

    def handle_inspect_data():
        if not hasattr(data_page, 'df') or data_page.df is None:
            update_status("⚠️ No data loaded.", "red")
            return

        update_status("🔍 Inspecting data...", "blue")
        clear_content()
        buffer = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer
        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            try:
                inspect_data(data_page.df)
            finally:
                sys.stdout = sys_stdout

        text_frame = tk.Frame(content_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        x_scroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        y_scroll = tk.Scrollbar(text_frame, orient=tk.VERTICAL)

        text_widget = tk.Text(
            text_frame, wrap="none",
            font=("Courier", 10),
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set
        )
        x_scroll.config(command=text_widget.xview)
        y_scroll.config(command=text_widget.yview)

        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        text_widget.insert(tk.END, buffer.getvalue())
        text_widget.config(state="disabled")  # Make read-only
        update_status("✅ Inspecting done", "green")
        split_btn.grid(row=0, column=2, sticky="ew", padx=5)
        split_btn.lift()

    def handle_split_data():
        if not hasattr(data_page, 'df') or data_page.df is None:
            update_status("⚠️ No data loaded.", "red")
            return

        try:
            x, y = split_features_target(data_page.df)
            data_page.x = x
            data_page.y = y

            update_status("✅ Split successful.", "green")
            clear_content()

            summary = f"✅ Features shape: {x.shape}\n✅ Target shape: {y.shape}"

            text_widget = tk.Text(content_frame, wrap="word", font=("Courier", 10))
            text_widget.insert(tk.END, summary)
            text_widget.config(state="disabled")
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            train_btn.grid(row=0, column=3, sticky="ew", padx=5)
            train_btn.lift()

        except Exception as e:
            update_status(f"❌ Split failed: {e}", "red")

    def handle_train_model():
        def _train():
            if not hasattr(data_page, 'x') or not hasattr(data_page, 'y'):
                update_status("⚠️ Please split the data first.", "red")
                return
    
            try:
                update_status("🚀 Training model...", "orange")
                clear_content()
    
                # === Create text widget early ===
                text_widget = tk.Text(content_frame, wrap="word", font=("Courier", 10))
                text_widget.pack(fill=tk.BOTH, expand=True)
    
                # Split data
                x_train, x_test, y_train, y_test = train_test_split(
                    data_page.x, data_page.y, test_size=0.2, random_state=42
                )
    
                n_samples, n_features = x_train.shape
                class_distribution = y_train.value_counts().to_dict()
    
                numerical_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    
                # Build model
                model = build_model_pipeline(numerical_cols, categorical_cols)
    
                # === STEP 1: Show setup info immediately ===
                setup_info = (
                    "🔧 Model Pipeline:\n"
                    "- Preprocessing:\n"
                    f"  • Numerical: {numerical_cols} (Imputer: mean, Scaler: StandardScaler)\n"
                    f"  • Categorical: {categorical_cols} (Imputer: 'missing', Encoder: OneHot)\n"
                    "- Classifier: RandomForestClassifier(n_estimators=100, random_state=42)\n\n"
                    f"📊 Training Data: {n_samples} samples, {n_features} features\n"
                    f"📈 Target Class Distribution: {class_distribution}\n"
                    "\n⏳ Training in progress...\n\n"
                )
    
                text_widget.insert(tk.END, setup_info)
                text_widget.update_idletasks()  # Forces GUI to refresh now
    
                # === STEP 2: Train the model ===
                trained_model = train_model(model, x_train, y_train)
    
                acc, report = evaluate_model(trained_model, x_test, y_test)
                save_model(trained_model)
    
                # === STEP 3: Append final results ===
                results_info = (
                    f"✅ Accuracy: {acc:.4f}\n\n"
                    f"📊 Classification Report:\n{report}"
                )
    
                text_widget.insert(tk.END, results_info)
                text_widget.config(state="disabled")
    
                data_page.last_report_t = setup_info + results_info
                data_page.last_report = report
    
                update_status(f"✅ Model trained. Accuracy: {acc:.2f}", "green")
                export_btn.grid(row=0, column=4, sticky="ew", padx=5)
                export_btn.lift()
    
            except Exception as e:
                update_status(f"❌ Training failed: {e}", "red")
    
        # Run training in a background thread to keep GUI responsive
        threading.Thread(target=_train, daemon=True).start()

    def handle_export_report():
        if not data_page.last_report_t:
            update_status("⚠️ No report to export.", "red")
            return

        try:
            clear_content()
            filename ="Training_report.txt"
            save_report(
                pd.DataFrame(),
                filename=filename,
                base_dir=BASE_DIR,
                title="Classification Report",
                description="Generated after model training"
            )
            update_status("✅ Report exported successfully.", "green")
        except Exception as e:
            update_status(f"❌ Failed to export: {e}", "red")

    data_page.last_report_t = None
    load_btn = tk.Button(
        button_frame,
        text="Load & Clean Data",
        command=handle_load_data,
        bg=BUTTON_COLOR, fg="white",
        padx=10, pady=5,
        font=(FONT_FAMILY, FONT_SIZE, "bold")
    )
    load_btn.grid(row=0, column=0, sticky="ew", padx=5)
    inspect_btn = tk.Button(
        button_frame,
        text="Inspect Data",
        command=handle_inspect_data,
        bg=BUTTON_COLOR, fg="white",
        padx=10, pady=5,
        font=(FONT_FAMILY, FONT_SIZE, "bold")
    )
    split_btn = tk.Button(
        button_frame,
        text="Split X & y",
        command=handle_split_data,
        bg=BUTTON_COLOR, fg="white",
        padx=10, pady=5,
        font=(FONT_FAMILY, FONT_SIZE, "bold")
    )
    train_btn = tk.Button(
        button_frame,
        text="Train Model",
        command=handle_train_model,
        bg=BUTTON_COLOR, fg="white",
        padx=10, pady=5,
        font=(FONT_FAMILY, FONT_SIZE, "bold")
    )
    export_btn = tk.Button(
        button_frame,
        text="Export Report",
        command=handle_export_report,
        bg=REPORT_BUTTON_COLOR, fg="black",
        padx=10, pady=5,
        font=(FONT_FAMILY, FONT_SIZE, "bold")
    )

def visualization_page(page_frame):
    """Render the visualization page where users can load data and generate plots."""
    visualization_page_frame = tk.Frame(page_frame, bg=BG_COLOR)
    visualization_page_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
    visualization_page_frame.columnconfigure(0, weight=1)
    visualization_page_frame.rowconfigure(1, weight=1)

    visualization_frame = tk.Frame(visualization_page_frame, bg=BG_COLOR)
    visualization_frame.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
    visualization_frame.columnconfigure((0, 1, 2), weight=1)

    status_label = tk.Label(
        visualization_page_frame,
        text="",
        bg=BG_COLOR,
        fg="green",
        font=(FONT_FAMILY, int(FONT_SIZE))
    )
    status_label.grid(row=1, column=0, pady=10, sticky="w")

    visualization_page.img_label = None

    def update_status(text, color="green"):
        status_label.config(text=text, fg=color)
        status_label.update_idletasks()

    def handle_load_data():
        update_status("🔄 Loading and cleaning data...", "orange")
        try:
            df = load_and_clean_data(CSV_PATH)
            visualization_page.df = df
            update_status(f"✅ Loaded {len(df)} rows.", "green")
        except Exception as e:
            update_status(f"❌ Failed to load data: {e}", "red")

    load_data_button = tk.Button(
        visualization_frame,
        text="Load & Clean Data",
        command=handle_load_data,
        bg=BUTTON_COLOR,
        fg="white",
        padx=10,
        pady=5,
        font=(FONT_FAMILY, int(FONT_SIZE), 'bold')
    )
    load_data_button.grid(row=0, column=0, padx=5, sticky="ew")

    plot_var = tk.StringVar()
    plot_var.set("Select Plot")

    plot_menu = ttk.OptionMenu(
        visualization_frame,
        plot_var,
        "Select Plot",
        "Age Distribution",
        "Hours Per Week Distribution",
        "Education Level Pie",
        "Workclass Distribution",
        "Income Distribution",
        "Age vs. Income",
        "Marital Status Distribution",
        "Occupation Treemap",
        "Race Pie",
        "Gender Distribution",
        "Education vs. Income",
        "Occupation vs. Income Heatmap",
        "Correlation Heatmap"
    )
    plot_menu.grid(row=0, column=1, padx=5, sticky="ew")

    def handle_plot_selection():
        selected_plot = plot_var.get()

        if not hasattr(visualization_page, 'df'):
            messagebox.showwarning("Data Not Loaded", "Please load data first.")
            return

        df = visualization_page.df
        output_map = {
            "Age Distribution": ("age_distribution.png",
                                 plot_age_distribution),
            "Hours Per Week Distribution": ("hours_per_week_distribution.png",
                                            plot_hours_per_week_distribution),
            "Education Level Pie": ("education_level_pie.png",
                                      plot_education_level_pie),
            "Workclass Distribution": ("workclass_distribution.png",
                                       plot_workclass_distribution),
            "Income Distribution": ("income_distribution.png",
                                    plot_income_distribution),
            "Age vs. Income": ("age_vs_income.png",
                               plot_age_vs_income),
            "Marital Status Distribution": ("marital_status_distribution.png",
                                            plot_marital_status_distribution),
            "Occupation Treemap": ("occupation_treemap.png",
                                        plot_occupation_treemap),
            "Race Pie": ("race_pie.png",
                                  plot_race_pie),
            "Gender Distribution": ("gender_distribution.png",
                                    plot_gender_distribution),
            "Education vs. Income": ("education_vs_income.png",
                                     plot_education_vs_income),
            "Occupation vs. Income Heatmap": ("occupation_vs_income_heatmap.png",
                                      plot_occupation_vs_income_heatmap),
            "Correlation Heatmap": ("correlation_heatmap.png",
                                    plot_correlation_heatmap),
        }

        if selected_plot not in output_map:
            messagebox.showwarning("Invalid Plot", "Please select a valid plot.")
            return

        filename, plot_func = output_map[selected_plot]
        try:
            plot_func(df, BASE_DIR)
            img_path = os.path.join(BASE_DIR, "Graphics", filename)

            if os.path.exists(img_path):
                pil_image = Image.open(img_path)
                pil_image.thumbnail((800, 500), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(pil_image)

                if visualization_page.img_label:
                    visualization_page.img_label.configure(image=img)
                    visualization_page.img_label.image = img
                else:
                    visualization_page.img_label = tk.Label(
                        visualization_page_frame, image=img, bg=BG_COLOR
                    )
                    visualization_page.img_label.image = img
                    visualization_page.img_label.grid(row=2, column=0, pady=10)

                update_status(f"✅ Plot displayed: {filename}")
            else:
                messagebox.showerror("Missing File", f"Plot not found: {filename}")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot:\n{e}")

    generate_button = tk.Button(
        visualization_frame,
        text="Generate Plot",
        command=handle_plot_selection,
        bg=BUTTON_COLOR,
        fg="white",
        padx=10,
        pady=5,
        font=(FONT_FAMILY, int(FONT_SIZE), 'bold')
    )
    generate_button.grid(row=0, column=2, padx=5, sticky="ew")

def configuration_page(page_frame):
    """Render the configuration page for adjusting interface preferences."""
    configuration_page_frame = tk.Frame(page_frame, bg=BG_COLOR)
    configuration_page_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)

    config = load_config()
    interface = config['interface']

    fonts = ["Arial", "Calibri", "Times New Roman", "Verdana", "Courier New"]
    colors = ["blue", "red", "cyan", "yellow", "green", "pink", "purple","grey"]
    font_sizes = ["12", "13", "14", "15", "16"]
    bg_color_choices = {
        "White": "white",
        "Beige": "beige",
        "Light Beige": "#f5f5dc",
        "Light Lavender": "#e6e6fa",
        "Powder Blue": "#b0e0e6",
        "Alice Blue": "#f0f8ff",
        "Honeydew": "#f0fff0",
        "Seashell": "#fff5ee",
        "Light Cyan": "#e0ffff",
        "Misty Rose": "#ffe4e1"
    }

    entries = {}
    row = 0

    def add_field(label_text, key, options=None, default="", is_dropdown=True):
        nonlocal row
        label = tk.Label(
            configuration_page_frame,
            text=label_text,
            bg=BG_COLOR,
            anchor="w"
        )
        label.grid(row=row, column=0, sticky="w", padx=(10, 5), pady=8)

        if key == "bg_color":
            hex_value = interface.get(key, default)
            name_value = next(
                (name for name, code in bg_color_choices.items() if code == hex_value),
                default
            )
            var = tk.StringVar(value=name_value)
        else:
            var = tk.StringVar(value=interface.get(key, default))

        if options:
            widget = ttk.Combobox(
                configuration_page_frame,
                textvariable=var,
                values=options,
                state="readonly"
            )
        else:
            widget = ttk.Entry(configuration_page_frame, textvariable=var)

        widget.grid(row=row, column=1, sticky="ew", padx=(5, 20), pady=8)
        entries[key] = var

        row += 1

    # Add fields (excluding "Sidebar Font Family" and "Sidebar Color")
    add_field("Font Family", "font_family", fonts, "Arial")
    add_field("Font Size", "font_size", font_sizes, "14")
    add_field("Report Button Color", "report_button_color", colors, "yellow")
    add_field("Export Button Color", "export_button_color", colors, "green")
    add_field("Button Color", "button_color", colors, "blue")
    add_field("Background Color", "bg_color", list(bg_color_choices.keys()), "White")

    def on_save():
        for key, var in entries.items():
            value = var.get()
            if key == "bg_color":
                interface[key] = bg_color_choices.get(value, value)
            else:
                interface[key] = value

        save_config(config)

        messagebox.showinfo(
            "Restart Required",
            "Configuration saved. The application will now restart."
        )

        python = sys.executable
        os.execl(python, python, *sys.argv)

    save_btn = tk.Button(
        configuration_page_frame,
        text="Save Configuration",
        command=on_save,
        bg=BUTTON_COLOR,
        fg="white",
        padx=10,
        pady=5,
        font=(FONT_FAMILY, int(FONT_SIZE), 'bold')
    )
    save_btn.grid(row=row, column=0, columnspan=2, pady=(25, 0))

    configuration_page_frame.columnconfigure(0, weight=1, minsize=150)
    configuration_page_frame.columnconfigure(1, weight=2, minsize=200)

def launch_gui():
    """Launch the Graphical User Interface"""
    apply_config()
    root = tk.Tk()
    root.geometry('900x600')
    root.title('Tkinter Income Predictor')

    toggle_icon = tk.PhotoImage(file=resource_path('images/open_menu.png'))
    close_icon = tk.PhotoImage(file=resource_path('images/close_menu.png'))
    data_icon = tk.PhotoImage(file=resource_path('images/data.png'))
    visualization_icon = tk.PhotoImage(file=resource_path('images/visualization.png'))
    config_icon = tk.PhotoImage(file=resource_path('images/configuration.png'))

    def switcher(ind, page, pg):
        """Switch between sidebar buttons and load respective page."""
        data_button_ind.config(bg=BUTTON_COLOR)
        visualization_button_ind.config(bg=BUTTON_COLOR)
        configuration_button_ind.config(bg=BUTTON_COLOR)
        ind.config(bg='white')
        if menu_sidebar_frame.winfo_width() > 50:
            shrink_menu_sidebar()

        for frame in page_frame.winfo_children():
            frame.destroy()

        page(pg)


    def expand_menu():
        """Animate expanding the sidebar."""
        current_width = menu_sidebar_frame.winfo_width()
        if current_width != 250:
            current_width += 10
            menu_sidebar_frame.config(width=current_width)
            root.after(ms=10, func=expand_menu)


    def expand_menu_sidebar():
        """Trigger sidebar expansion."""
        expand_menu()
        toggle_button.config(image=close_icon)
        toggle_button.config(command=shrink_menu_sidebar)


    def shrink_menu():
        """Animate shrinking the sidebar."""
        current_width = menu_sidebar_frame.winfo_width()
        if current_width != 50:
            current_width -= 10
            menu_sidebar_frame.config(width=current_width)
            root.after(ms=12, func=shrink_menu)


    def shrink_menu_sidebar():
        """Trigger sidebar shrinking."""
        shrink_menu()
        toggle_button.config(image=toggle_icon)
        toggle_button.config(command=expand_menu_sidebar)


    # frames
    page_frame = tk.Frame(root, bg=BG_COLOR)
    page_frame.place(relwidth=1.0, relheight=1.0, x=30)
    data_page(page_frame)

    menu_sidebar_frame = tk.Frame(root, bg=BUTTON_COLOR)

    # buttons
    toggle_button = tk.Button(
        menu_sidebar_frame, image=toggle_icon, bg=BUTTON_COLOR, bd=0,
        activebackground=BUTTON_COLOR, command=lambda: expand_menu_sidebar()
    )
    toggle_button.place(x=5, y=10, width=40, height=40)

    data_button = tk.Button(
        menu_sidebar_frame, image=data_icon, bg=BUTTON_COLOR, bd=0,
        activebackground=BUTTON_COLOR,
        command=lambda: switcher(ind=data_button_ind, page=data_page,
                                 pg=page_frame)
    )
    data_button.place(x=5, y=140, width=40, height=40)

    data_button_ind = tk.Label(menu_sidebar_frame, bg='white')
    data_button_ind.place(x=2, y=140, width=3, height=40)

    data_button_title = tk.Label(
        menu_sidebar_frame, text='Data manipulation', bg=BUTTON_COLOR, fg='white',
        font=(FONT_FAMILY, int(SIDEBAR_FONT_SIZE), 'bold'), anchor=tk.W
    )
    data_button_title.place(x=50, y=140, width=180, height=40)
    data_button_title.bind('<Button-1>', lambda e: switcher(ind=data_button_ind,
                                                            page=data_page,
                                                            pg=page_frame)
                           )

    visualization_button = tk.Button(
        menu_sidebar_frame, image=visualization_icon, bg=BUTTON_COLOR, bd=0,
        activebackground=BUTTON_COLOR,
        command=lambda: switcher(ind=visualization_button_ind,
                                 page=visualization_page,
                                 pg=page_frame)
    )
    visualization_button.place(x=5, y=220, width=40, height=40)

    visualization_button_ind = tk.Label(menu_sidebar_frame, bg=BUTTON_COLOR)
    visualization_button_ind.place(x=2, y=220, width=3, height=40)

    visualization_button_title = tk.Label(
        menu_sidebar_frame, text='Visualizations', bg=BUTTON_COLOR, fg='white',
        font=(FONT_FAMILY, int(SIDEBAR_FONT_SIZE), 'bold'), anchor=tk.W
    )
    visualization_button_title.place(x=50, y=220, width=180, height=40)
    visualization_button_title.bind(
        '<Button-1>', lambda e: switcher(ind=visualization_button_ind,
                                         page=visualization_page,
                                         pg=page_frame)
    )

    configuration_button = tk.Button(
        menu_sidebar_frame, image=config_icon, bg=BUTTON_COLOR, bd=0,
        activebackground=BUTTON_COLOR,
        command=lambda: switcher(ind=configuration_button_ind,
                                 page=configuration_page,
                                 pg=page_frame)
    )
    configuration_button.place(x=5, y=300, width=40, height=40)

    configuration_button_ind = tk.Label(menu_sidebar_frame, bg=BUTTON_COLOR)
    configuration_button_ind.place(x=2, y=300, width=3, height=40)

    configuration_button_title = tk.Label(
        menu_sidebar_frame, text='Configuration', bg=BUTTON_COLOR, fg='white',
        font=(FONT_FAMILY, int(SIDEBAR_FONT_SIZE), 'bold'), anchor=tk.W
    )
    configuration_button_title.place(x=50, y=300, width=180, height=40)
    configuration_button_title.bind(
        '<Button-1>', lambda e: switcher(ind=configuration_button_ind,
                                         page=configuration_page,
                                         pg=page_frame)
    )

    menu_sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=3)
    menu_sidebar_frame.pack_propagate(flag=False)
    menu_sidebar_frame.configure(width=50)

    try:
        root.mainloop()
    except Exception:
        pass