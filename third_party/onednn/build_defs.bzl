def onednn_deps():
    """Shorthand for select() to pull in the correct set of oneDNN library deps.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return ["@onednn_gpu//:onednn_gpu"]
