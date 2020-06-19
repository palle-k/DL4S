import lldb

def describe_tensor(value, internal_dict):
    stream = lldb.SBStream()
    value.GetExpressionPath(stream)
    expr = stream.GetData() + ".debugDescription"
    val = value.GetTarget().EvaluateExpression(expr)
    return str(val.GetSummary()).strip().strip('"').replace("\\n", "\n")
    

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("type summary add -F " + __name__ + '.describe_tensor -x "DL4S\\.Tensor<.+>" -w swift')
    debugger.HandleCommand("type summary add -F " + __name__ + '.describe_tensor -x "DL4S\\.ShapedBuffer<.+>" -w swift')

