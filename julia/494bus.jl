using GenericMessagePassing
using CSV, DataFrames

using Graphs, GraphIO
using TensorInference, OMEinsum

g = loadgraph("data/494bus.dot")
df_J = CSV.read("data/494bus_J_random.csv", DataFrame)
df_h = CSV.read("data/494bus_h_random.csv", DataFrame)

Js_dict = Dict()
for row in eachrow(df_J)
    Js_dict[minmax(row.v1 + 1, row.v2 + 1)] = row.J
end

hs = df_h.h 
Js = Float64[]
for e in edges(g)
    v1, v2 = src(e), dst(e)
    push!(Js, Js_dict[minmax(v1, v2)])
end

tn, code, tensors = GenericMessagePassing.ising_model(g, -1 .* hs, -1 .* Js, 0.5)
ti_sol = marginals(tn)


bp_sol = marginal_tnbp(tn, TNBPConfig(verbose = true, r = 7, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100), error = 1e-6))

total_err = 0.0
for i in keys(bp_sol)
    total_err += sum((ti_sol[[i]] - bp_sol[i]).^2)
end

total_err = sqrt(total_err) / 494

@info "Total error: $total_err"