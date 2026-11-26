using Test, L2OSobolev
import Aqua, ExplicitImports, JET

@testset "Aqua" begin
    Aqua.test_all(L2OSobolev)
end

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(L2OSobolev) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(L2OSobolev) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(L2OSobolev) === nothing
end

@testset "JET" begin
    if pkgversion(JET) >= v"0.10"
        JET.test_package(L2OSobolev; target_modules = (L2OSobolev,))
    else
        JET.test_package(L2OSobolev; target_defined_modules = true)
    end
end
