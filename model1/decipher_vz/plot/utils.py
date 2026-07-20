def activate_journal_quality():
    """Activate journal quality settings for plotting."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # 1. Manually set the same parameters Scanpy would, but safely
    mpl.rcParams["figure.dpi"] = 100  # Display DPI
    mpl.rcParams["savefig.dpi"] = 400  # High-quality save DPI
    mpl.rcParams["font.size"] = 18
    
    # 2. Text handling for Adobe Illustrator
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # 3. Aesthetics
    plt.rcParams["axes.grid"] = False
    
    # 4. Handle the high-res "retina" display for Jupyter manually
    try:
        import IPython.display
        # This is the modern replacement for set_matplotlib_formats
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats('retina')
    except (ImportError, ModuleNotFoundError):
        pass