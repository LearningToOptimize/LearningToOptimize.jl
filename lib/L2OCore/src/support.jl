"""
    supports(method::AbstractL2OMethod, class::AbstractProblemClass) -> Bool

Whether `method` is defined for problems of `class`, i.e. whether the
mathematical assumptions of the method hold for that class. The fallback is
`false`: a method must opt in explicitly, class by class, e.g.

```julia
L2OCore.supports(::MyMethod, ::L2OCore.AbstractConvexClass) = true
```

There is deliberately no silent fallback that would run a method on a class
its derivation does not cover; see [`check_supports`](@ref).
"""
supports(::AbstractL2OMethod, ::AbstractProblemClass) = false

"""
    UnsupportedProblemError(method, class)

Thrown by [`check_supports`](@ref) when a method is used on a problem class it
does not support.
"""
struct UnsupportedProblemError <: Exception
    method::AbstractL2OMethod
    class::AbstractProblemClass
end

function Base.showerror(io::IO, e::UnsupportedProblemError)
    M = nameof(typeof(e.method))
    print(
        io,
        "UnsupportedProblemError: method `$M` does not support problems of class ",
        "`$(nameof(typeof(e.class)))`: it is only defined for classes where ",
        "`supports(::$M, class)` is true. ",
        "If the problem was mis-detected, pass `class = ...` to `ParametricProblem`; ",
        "otherwise choose a method that supports this class.",
    )
end

"""
    check_supports(problem::ParametricProblem, method::AbstractL2OMethod)

Throw an [`UnsupportedProblemError`](@ref) unless
`supports(method, problem_class(problem))`. Called by [`L2OPipeline`](@ref)
before anything else.
"""
function check_supports(problem::AbstractParametricProblem, method::AbstractL2OMethod)
    supports(method, problem_class(problem)) ||
        throw(UnsupportedProblemError(method, problem_class(problem)))
    return nothing
end
