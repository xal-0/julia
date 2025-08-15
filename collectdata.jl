using DataFrames, CSV

configs = [
    nothing, "copy-0",
    "lz4-1", "lz4-2",
    "lz4hc-2", "lz4hc-9", "lz4hc-10", "lz4hc-12",
    "zstd-1", "zstd-5", "zstd-15", "zstd-22",
]

function compile(config)
    name = config !== nothing ? "hello-$config" : "hello-uncomp"
    e = copy(ENV)
    e["JULIA_IMAGE_THREADS"] = "1"
    e["JULIA_NUM_THREADS"] = "1"
    if config !== nothing
        e["JULIA_IMAGE_COMPRESSION"] = config
    end
    src = "/Users/user/c/utils/juliac-hello.jl"
    c = Cmd(`./usr/bin/julia contrib/juliac/juliac.jl --output-exe $name $src`; env=e)
    (; value, time) = @timed read(c, String)
    r = r"\[COMP\] done, ([0-9]+) B -> ([0-9]+) B, ratio [0-9.]+% \(([0-9]+) ms\)"
    heapsize_uncomp, heapsize_comp, comp_time = match(r, value)
    total_size = filesize(name)
    path = "./$name"
    for i=1:3 run(`$path`) end  # warmup
    N = 10
    runtime = (@timed for i=1:N run(`$path`) end).time / N
    heapsize_uncomp, heapsize_comp, comp_time = parse(Int, heapsize_uncomp), parse(Int, heapsize_comp), parse(Int, comp_time) / 1000
    kind = "uncomp"
    level = 0
    if config !== nothing
        kind, level = split(config, "-")
        level = parse(Int, level)
    end
    return (; kind, level, time, heapsize_uncomp, heapsize_comp, comp_time, total_size, runtime)
end

S = Base.Semaphore(4)

function compile_task(config)
    Base.acquire(S) do
        println("compiling $config")
        compile(config)
    end
end

tasks = [Threads.@spawn compile_task(c) for c in configs]
data = [fetch(t) for t in tasks]
CSV.write("out.csv", DataFrame(data))
