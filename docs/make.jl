using Documenter
using ParallelParticleSwarms

makedocs(;
    modules = [ParallelParticleSwarms],
    checkdocs = :exports,
    sitename = "ParallelParticleSwarms.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/ParallelParticleSwarms/stable/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/ParallelParticleSwarms.jl",
    devbranch = "main",
)
