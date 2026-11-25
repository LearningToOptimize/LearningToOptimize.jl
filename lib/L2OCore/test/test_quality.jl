using Test, L2OCore
import Aqua, ExplicitImports, JET

@testset "Aqua" begin
    Aqua.test_all(L2OCore)
end

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(L2OCore) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(L2OCore) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(L2OCore) === nothing
end

@testset "JET" begin
    if pkgversion(JET) >= v"0.10"
        JET.test_package(L2OCore; target_modules = (L2OCore,))
    else
        JET.test_package(L2OCore; target_defined_modules = true)
    end
end
