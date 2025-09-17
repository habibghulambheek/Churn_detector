from tkinter import *
from tkinter import ttk, filedialog
import xgboost as xgb
import numpy as np
import os
import csv

# ----------------- Load Model -----------------
curr_dir = os.getcwd()
model_path = os.path.join(curr_dir, "xgb.json")
icon_path = os.path.join(curr_dir, "signal-tower.ico")

model = xgb.XGBClassifier()
model.load_model(model_path)

# ----------------- Colors & Fonts -----------------
BG = "#F5F7FA"
CARD_BG = "white"
ACCENT = "#0E5B88"
GOOD = "#2E7D32"
BAD = "#B71C1C"
WARN = "#E65100"
FONT = ("Segoe UI", 10)
TITLE_FONT = ("Segoe UI", 26, "bold")

# ----------------- Root Window -----------------
root = Tk()
root.configure(bg=BG)
root.attributes('-fullscreen', True)
root.title("Churn Detector")
root.geometry("820x720")
root.iconbitmap(icon_path)
root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
root.bind('<F11>', lambda e: root.attributes('-fullscreen', True))
# ----------------- Scrollable area (Canvas + Scrollbar) -----------------
container = Frame(root, bg=BG)
container.pack(fill="both", expand=True, padx=10, pady=8)

canvas = Canvas(container, bg=BG, highlightthickness=0)
v_scroll = ttk.Scrollbar(container, orient = 'vertical', command = canvas.yview)
canvas.configure(yscrollcommand= v_scroll.set)

v_scroll.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

mainFrame = Frame(canvas, bg=CARD_BG, bd=2, relief="groove")
mainFrame_id = canvas.create_window((0, 0), window=mainFrame, anchor="nw", width=780)

def on_frame_configure(event = None):
    canvas.configure(scrollregion= canvas.bbox('all'))
    canvas.itemconfig(mainFrame_id, width = canvas.winfo_width() - 4)
mainFrame.bind('<Configure>', on_frame_configure)

def _on_mousewheel(event):
    if event.num == 5 or event.delta < 0:
        canvas.yview_scroll(1, 'units')
    elif event.num == 4 or event.delta > 0:
        canvas.yview_scroll(-1, 'units')
canvas.bind_all('<MouseWheel>', _on_mousewheel)
canvas.bind_all('Button-4', _on_mousewheel)
canvas.bind_all('Button-5', _on_mousewheel)

# ----------------- Title -----------------
Label(mainFrame, text="📊 Customer Churn Detector", font=TITLE_FONT,
      pady=12, fg=ACCENT, bg=CARD_BG).pack(anchor="n", pady=(12, 6))

# ----------------- Data Structures -----------------
discrete_features = {
    'gender': ("Male", "Female"),
    'SeniorCitizen': ("No", "Yes"),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("No", "Yes"),
    'MultipleLines': ("No", "Yes", "No phone service"),
    'InternetService': ('No', 'DSL', 'Fiber optic'),
    'OnlineSecurity': ('No', 'Yes', 'No internet service'),
    'OnlineBackup': ('No', 'Yes', 'No internet service'),
    'DeviceProtection': ('No', 'Yes', 'No internet service'),
    'TechSupport': ('No', 'Yes', 'No internet service'),
    'StreamingTV': ('No', 'Yes', 'No internet service'),
    'StreamingMovies': ('No', 'Yes', 'No internet service'),
    'Contract': ("Month-to-month", 'One year', 'Two year'),
    'PaperlessBilling': ("No", 'Yes'),
    'PaymentMethod': ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")
}

continuous_features = ["tenure", "MonthlyCharges", "TotalCharges"]

all_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
'SeniorCitizen_1', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
'MultipleLines_No phone service', 'MultipleLines_Yes',
'InternetService_Fiber optic', 'InternetService_No',
'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
'OnlineBackup_No internet service', 'OnlineBackup_Yes',
'DeviceProtection_No internet service', 'DeviceProtection_Yes',
'TechSupport_No internet service', 'TechSupport_Yes',
'StreamingTV_No internet service', 'StreamingTV_Yes',
'StreamingMovies_No internet service', 'StreamingMovies_Yes',
'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
'PaymentMethod_Credit card (automatic)',
'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# ----------------- Helpers -----------------
def create_dropdown(frame, label_text, options, row, col):
    Label(frame, text=label_text, font=("Segoe UI", 11, "bold"),
          bg=CARD_BG, fg=ACCENT).grid(row=row, column=col, pady=(6,0), sticky="w")
    var = StringVar(value=options[0])
    dropdown = ttk.Combobox(frame, values=options, textvariable=var, font=FONT, state="readonly", width=22)
    dropdown.grid(row=row+1, column=col, padx=8, pady=6, sticky="w")
    return var

def create_entry(frame, label_text, col):
    Label(frame, text=label_text, font=("Segoe UI", 11, "bold"),
          bg=CARD_BG, fg=ACCENT).grid(row=0, column=col, pady=(6,0), sticky="w")
    entry = Entry(frame, font=FONT, width=18, bd=2, relief="solid")
    entry.grid(row=1, column=col, padx=8, pady=6, sticky="w")
    return entry

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    def show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip = Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = Label(self.tip, text=self.text, justify="left", background="#FFF3E0", relief="solid", borderwidth=1, font=("Segoe UI", 9))
        label.pack(ipadx=6, ipady=3)
    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

def on_enter(e):
    e.widget.config(bg="#0B4F73")
def on_leave(e):
    e.widget.config(bg=ACCENT)

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

# ----------------- Input Layout -----------------
discrete_frame = LabelFrame(mainFrame, text="Customer Information", font=("Segoe UI", 12, "bold"),
                            bg=CARD_BG, fg=ACCENT, padx=14, pady=10, labelanchor="n")
discrete_frame.pack(padx=16, pady=(6,12), fill="x")

dropdown_vars, Entry_vars = {}, {}

row, col = 0, 0
for feature, options in discrete_features.items():
    dropdown_vars[feature] = create_dropdown(discrete_frame, feature, options, row, col)
    col += 1
    if col == 3:
        row += 2
        col = 0

cont_frame = LabelFrame(mainFrame, text="Numerical Features", font=("Segoe UI", 12, "bold"),
                        bg=CARD_BG, fg=ACCENT, padx=14, pady=10, labelanchor="n")
cont_frame.pack(padx=16, pady=(6,12), fill="x")

# Slider params with resolution for floats
slider_params = {
    "tenure": {"from_": 0.0, "to": 105.0, "resolution": 1.0, "format": lambda v: str(int(round(v)))},
    "MonthlyCharges": {"from_": 0.0, "to": 200.0, "resolution": 0.01, "format": lambda v: f"{v:.2f}"},
    "TotalCharges": {"from_": 0.0, "to": 20000.0, "resolution": 0.01, "format": lambda v: f"{v:.2f}"}
}

def make_slider_row(frame, label_text, col):
    Label(frame, text=label_text, font=("Segoe UI", 11, "bold"), bg=CARD_BG, fg=ACCENT).grid(row=0, column=col, sticky="w", padx=8)
    entry = Entry(frame, font=FONT, width=12, bd=2, relief="solid")
    entry.grid(row=1, column=col, padx=8, pady=6, sticky="w")
    params = slider_params[label_text]
    var = DoubleVar()
    scale = Scale(frame, variable=var, orient="horizontal", from_=params["from_"], to=params["to"],
                  resolution=params["resolution"], length=300, showvalue=0)
    scale.grid(row=1, column=col+1, padx=(4,16), pady=6, sticky="w")
    def var_changed(*_):
        v = var.get()
        entry.delete(0, END)
        entry.insert(0, params["format"](v))
    var.trace_add("write", var_changed)
    def entry_to_var(e=None):
        val = entry.get().strip()
        if is_float(val):
            v = float(val)
            if v < params["from_"]:
                v = params["from_"]
            if v > params["to"]:
                v = params["to"]
            var.set(v)
        else:
            var_changed()
    entry.bind("<FocusOut>", entry_to_var)
    entry.bind("<Return>", entry_to_var)
    if label_text == "tenure":
        var.set(12.0)
    elif label_text == "MonthlyCharges":
        var.set(50.0)
    else:
        var.set(600.0)
    var_changed()
    return entry, scale, var

tenure_entry, tenure_scale, tenure_var = make_slider_row(cont_frame, "tenure", 0)
Monthly_entry, Monthly_scale, Monthly_var = make_slider_row(cont_frame, "MonthlyCharges", 2)
Total_entry, Total_scale, Total_var = make_slider_row(cont_frame, "TotalCharges", 4)

Entry_vars["tenure"] = tenure_entry
Entry_vars["MonthlyCharges"] = Monthly_entry
Entry_vars["TotalCharges"] = Total_entry

Label(mainFrame, text="Tip: Use mouse wheel to scroll. Hover buttons for hints.", font=("Segoe UI", 9), bg=CARD_BG, fg="#666").pack(anchor="w", padx=20, pady=(0,8))

# ----------------- Actions -----------------
result_frame = Frame(mainFrame, bg=CARD_BG, bd=1, relief="solid", padx=12, pady=12)
result_frame.pack(fill="x", padx=16, pady=(6,12))

result_lbl = Label(result_frame, text='', font=("Segoe UI", 12, "bold"),
                   bg=CARD_BG, fg=ACCENT, anchor="w", justify="left")
result_lbl.pack(fill="x", pady=(2,8))

progress = ttk.Progressbar(result_frame, orient="horizontal", length=600, mode="determinate")
progress.pack(fill="x", pady=(0,6))

details_lbl = Label(result_frame, text='Probability will show here after prediction.', font=("Segoe UI", 9), bg=CARD_BG, fg="#444")
details_lbl.pack(anchor="w")

def build_feature_vector():
    X_dict = {cols: 0 for cols in all_cols}
    for feature in continuous_features:
        val = Entry_vars[feature].get()
        if not is_float(val):
            raise ValueError(f"{feature} must be numeric.")
        X_dict[feature] = float(val)
    for feature in discrete_features:
        if feature == 'SeniorCitizen':
            val = dropdown_vars[feature].get()
            X_dict['SeniorCitizen_1'] = 1 if val in ['Yes', '1', '1.0', 'yes', 'True', 'true'] else 0
        else:
            col_name = f"{feature}_{dropdown_vars[feature].get()}"
            if col_name in X_dict:
                X_dict[col_name] = 1
            if feature == 'gender' and dropdown_vars[feature].get() == 'Male':
                X_dict['gender_Male'] = 1
    return np.array(list(X_dict.values())).reshape(1, -1)

def show_result():
    try:
        X = build_feature_vector()
        prob_y = model.predict_proba(X)[0][1] * 100
        prob_y = round(prob_y, 2)
        progress['value'] = prob_y
        details_lbl.config(text=f"Churn probability: {prob_y}%")
        if prob_y < 50:
            result_lbl.config(text=f"No Churn Detected ✅  ({prob_y}%)", fg=GOOD, bg=CARD_BG)
        else:
            result_lbl.config(text=f"⚠️ Churn Detected!  ({prob_y}%)", fg=BAD, bg=CARD_BG)
    except Exception as e:
        result_lbl.config(text=f"Error: {e}", fg=WARN, bg=CARD_BG)
        progress['value'] = 0
        details_lbl.config(text="Please correct inputs and try again.")

def reset_form():
    for k, v in discrete_features.items():
        dropdown_vars[k].set(v[0])
    tenure_var.set(12.0)
    Monthly_var.set(50.0)
    Total_var.set(600.0)
    result_lbl.config(text='', fg=ACCENT)
    progress['value'] = 0
    details_lbl.config(text='Form reset.')

# ----------------- Upload: parse EXACT order specified by user -----------------
# Expected order:
# gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,
# OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,
# PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges

expected_order = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService',
                  'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
                  'PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']

def upload_data():
    file_loc = filedialog.askopenfilename(initialdir=curr_dir, title="Select your record",
                                          filetypes=(("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")))
    if not file_loc:
        return
    try:
        with open(file_loc, 'r') as f:
            text = f.read().strip()
        # single-line CSV-style tokens
        tokens = [t.strip() for t in text.replace('\n', '').split(',') if t.strip() != '']
        if len(tokens) != len(expected_order):
            raise ValueError(f"Expected at least {len(expected_order)} values in the file (found {len(tokens)}).")
        # take first len(expected_order) tokens (user said file follows that order)
        tokens = tokens[:len(expected_order)]
        mapping = dict(zip(expected_order, tokens))
        # map discrete
        for k in discrete_features.keys():
            if k in mapping:
                val = mapping[k]
                if k == 'SeniorCitizen':
                    dropdown_vars[k].set('Yes' if val in ['1', '1.0', 'Yes', 'yes', 'True', 'true'] else 'No')
                else:
                    # sometimes partner/dependents/phoneservice are '0'/'1' -> convert
                    if val in ['0', '0.0'] and 'No' in discrete_features[k]:
                        dropdown_vars[k].set('No')
                    elif val in ['1', '1.0'] and 'Yes' in discrete_features[k]:
                        dropdown_vars[k].set('Yes')
                    elif val in discrete_features[k]:
                        dropdown_vars[k].set(val)
        # map continuous
        for k in continuous_features:
            if k in mapping and is_float(mapping[k]):
                Entry_vars[k].delete(0, END)
                Entry_vars[k].insert(0, mapping[k])
        # sync slider vars to entries
        try:
            tenure_var.set(float(Entry_vars['tenure'].get()))
        except:
            tenure_var.set(12.0)
        try:
            Monthly_var.set(float(Entry_vars['MonthlyCharges'].get()))
        except:
            Monthly_var.set(50.0)
        try:
            Total_var.set(float(Entry_vars['TotalCharges'].get()))
        except:
            Total_var.set(600.0)

        result_lbl.config(text="✅ File loaded (order-based). Adjust values if needed.", fg=ACCENT)
    except Exception as e:
        result_lbl.config(text=f"⚠️ Invalid File\n{e}", fg=WARN)

# ----------------- Buttons -----------------
btn_frame = Frame(mainFrame, bg=CARD_BG)
btn_frame.pack(pady=6, padx=16, anchor="w", fill="x")

predict_btn = Button(btn_frame, text="🔮 Predict Churn", command=show_result,
       bg=ACCENT, fg="white", font=("Segoe UI", 12, "bold"),
       relief="flat", padx=18, pady=8)
predict_btn.grid(row=0, column=0, padx=(2,8))

upload_btn = Button(btn_frame, text="📂 Upload Data", command=upload_data,
       bg=ACCENT, fg="white", font=("Segoe UI", 12, "bold"),
       relief="flat", padx=18, pady=8)
upload_btn.grid(row=0, column=1, padx=(4,8))

reset_btn = Button(btn_frame, text="♻️ Reset", command=reset_form,
       bg="#6C757D", fg="white", font=("Segoe UI", 12, "bold"),
       relief="flat", padx=14, pady=8)
reset_btn.grid(row=0, column=2, padx=(4,8))

for b, tip in [(predict_btn, "Compute churn probability using the model"),
               (upload_btn, "Load a saved record (TXT in the specified order)"),
               (reset_btn, "Clear form and reset defaults")]:
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)
    ToolTip(b, tip)

# ----------------- Footer -----------------
footer = Frame(mainFrame, bg=CARD_BG)
footer.pack(fill="x", padx=16, pady=(10,18))
Label(footer, text="Made by Habib Ghulam Bheek❤️.", font=("Segoe UI", 9), bg=CARD_BG, fg="#666").pack(side="left")

root.update_idletasks()
on_frame_configure()

root.mainloop()
