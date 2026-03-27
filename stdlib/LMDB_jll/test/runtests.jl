# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, LMDB_jll

@testset "LMDB_jll" begin
    @test isfile(liblmdb_path)
end
