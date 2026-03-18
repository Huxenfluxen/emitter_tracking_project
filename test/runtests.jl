using Test
using DSP
include("../src/signal_generation/modulators.jl")
using .SignalGeneration

@testset "Signal Generation Physics Tests" begin
    
    @testset "DSSS Gold Codes" begin
        # A 7-bit LFSR should ALWAYS produce exactly 127 chips (2^7 - 1)
        code = generate_gold_code(7)
        @test length(code) == 127
        
        # Power should be normalized to ±1
        @test all(x -> x == 1.0 || x == -1.0, code)
    end

    @testset "RRC Filter Nyquist Properties" begin
        sps = 8
        span = 10
        h = SignalGeneration.create_rrc_filter(sps, 0.35, span)
        
        # The filter should be symmetric
        @test h == reverse(h)
        
        # The length should strictly follow the span and sps formula
        @test length(h) == (span * sps) + 1
    end
end