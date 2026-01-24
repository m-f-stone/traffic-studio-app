# studio/theme.py
from tkinter import ttk

def apply_theme_base(root):
    """
    Apply a consistent ttk theme across the application.
    Uses 'clam' as a robust base and normalizes paddings, tab spacing, and labelframe headers.
    """
    style = ttk.Style(root)
    # Choose a widely available theme
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Normalize paddings
    base_pad = {"padding": 4}
    for element in ("TLabel","TButton","TEntry","TCombobox","TCheckbutton","TRadiobutton","Treeview","TNotebook","TLabelframe","TLabelframe.Label"):
        try:
            style.configure(element, **base_pad)
        except Exception:
            pass

    # Notebook tabs: extra padding for visibility
    try:
        style.configure("TNotebook.Tab", padding=(12, 6))
    except Exception:
        pass

    # Labelframe title weight
    try:
        style.configure("TLabelframe.Label", padding=(6, 0))
    except Exception:
        pass
