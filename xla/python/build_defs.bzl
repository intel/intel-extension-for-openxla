def gen_xla_version(name, header_in, header_out, **kwargs):
    tool = "//xla/python:gen_xla_version"

    native.genrule(
        name = name,
        srcs = [header_in],
        outs = [header_out],
        tools = [tool],
        cmd = "$(location {}) ".format(tool) + "--in=$< " + "--out=$@",
        stamp = True,
        **kwargs
    )
