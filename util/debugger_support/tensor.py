import lldb

def describe_tensor(value, internal_dict):
    stream = lldb.SBStream()
    value.GetExpressionPath(stream)
    expr = stream.GetData() + ".debugDescription"
    
    val = value.GetTarget().EvaluateExpression(expr)
    if val.GetError().GetCString() is None:
        return str(val.GetSummary()).strip().strip('"').replace("\\n", "\n")
    
    val = value.GetFrame().EvaluateExpression(expr)
    if val.GetError().GetCString() is None:
        return str(val.GetSummary()).strip().strip('"').replace("\\n", "\n")
    
    val = value.CreateValueFromExpression("x", expr)
    if val.GetError().GetCString() is None:
        return str(val.GetSummary()).strip().strip('"').replace("\\n", "\n")
        
    return "could not generate description"

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("type summary add -F " + __name__ + '.describe_tensor -x "DL4S\\.Tensor<.+>" -w swift')
    debugger.HandleCommand("type summary add -F " + __name__ + '.describe_tensor -x "DL4S\\.ShapedBuffer<.+>" -w swift')

