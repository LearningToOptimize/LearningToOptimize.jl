using Documenter
using DocumenterMermaid
using Literate
using LearningToOptimize
using L2OCore

# Literate examples -> markdown, next to their sources.
const EXAMPLES = joinpath(@__DIR__, "src", "examples")
for file in readdir(EXAMPLES)
    endswith(file, ".jl") || continue
    Literate.markdown(joinpath(EXAMPLES, file), EXAMPLES; documenter = true, credit = false)
end

for mod in (LearningToOptimize, L2OCore)
    DocMeta.setdocmeta!(mod, :DocTestSetup, :(using $(nameof(mod))); recursive = true)
end

makedocs(;
    # One site for the main package and the lib/ subpackages: cross-references
    # between them resolve and there is a single search index.
    modules = [LearningToOptimize, L2OCore],
    sitename = "LearningToOptimize.jl",
    repo = Remotes.GitHub("LearningToOptimize", "LearningToOptimize.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://LearningToOptimize.github.io/LearningToOptimize.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 300 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Design" => "design.md",
        "Package guide" => [
            "Getting started" => "examples/penalty_lp.md",
            "Adding a method" => "guide/extending.md",
            "Dataset generation: file formats" => "arrow.md",
            "Dataset generation: parameter types" => "parametertype.md",
        ],
        "API reference" =>
            ["L2OCore" => "api_l2ocore.md", "LearningToOptimize" => "api.md"],
    ],
)

deploydocs(;
    repo = "github.com/LearningToOptimize/LearningToOptimize.jl",
    devbranch = "main",
    push_preview = true,
)
