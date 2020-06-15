import lldb

def describe_tensor(value, internal_dict):
    stream = lldb.SBStream()
    value.GetExpressionPath(stream)
    expr = "(" + stream.GetData() + ").description"
    val = value.CreateValueFromExpression("str", expr)
    return val.GetSummary()


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("type summary add -F " + __name__ + '.describe_tensor -x "DL4S\\.Tensor<.+>" -w swift')

