using DataFrames, CSV, CairoMakie

# uncomp,0,165.114162083,143185744,143185744,0.0,161410464,0.10597429169999999
data = CSV.read("out.csv", DataFrame)
data.ratio = data.heapsize_uncomp ./ data.heapsize_comp
data.comptimep = data.comp_time ./ data.time
mtypes = Dict("copy" => :circle, "zstd" => :rect, "lz4" => :utriangle, "lz4hc" => :dtriangle)
ctypes = Dict("copy" => :grey, "zstd" => :red, "lz4" => :blue, "lz4hc" => :pink)

markers = [mtypes[k] for k in data.kind]
colors = [ctypes[x] for x in data.kind]

f = Figure()
ax1 = Axis(f[1, 1], xlabel="Decompression and run time (ms)", ylabel="Compression ratio (heap only)", title="Decompression")
scatter!(ax1, data.runtime * 1000, data.ratio, marker=markers, markersize=12, color=colors,
         label=[k => (; marker=m, color=c) for (m, c, k) in zip(markers, colors, data.kind)],)
annotation!(ax1, data.runtime * 1000, data.ratio, text=[l == 0 ? "" : string(l) for l in data.level])
Legend(f[1,2], [
    [MarkerElement(color=ctypes[k], marker=mtypes[k])]
    for k in keys(mtypes)
], collect(keys(mtypes)))
save("decompression.png", f)


f = Figure()
ax1 = Axis(f[1, 1], xlabel="Percentage of compile time spent compressing", ylabel="Compression ratio (heap only)", title="Compression")
scatter!(ax1, data.comptimep * 100, data.ratio, marker=markers, markersize=12, color=colors,
         label=[k => (; marker=m, color=c) for (m, c, k) in zip(markers, colors, data.kind)],)
annotation!(ax1, data.comptimep * 100, data.ratio, text=[l == 0 ? "" : string(l) for l in data.level])
Legend(f[1,2], [
    [MarkerElement(color=ctypes[k], marker=mtypes[k])]
    for k in keys(mtypes)
], collect(keys(mtypes)))
save("compression.png", f)
