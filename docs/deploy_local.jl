#!/usr/bin/env julia
# Manual gh-pages deployment. Since v0.6.x the docs are NOT built in GitHub Actions
# (Documentation.yml was removed 2026-07-10 — remote compilation was too slow); this
# script builds the site locally and pushes it to gh-pages.
#
# Usage:
#   julia --project=docs docs/deploy_local.jl                    # build + deploy dev/
#   julia --project=docs docs/deploy_local.jl --version v0.6.0   # release: also updates vX.Y + stable symlinks and versions.js
#   julia --project=docs docs/deploy_local.jl --skip-build       # reuse existing docs/build
#   julia --project=docs docs/deploy_local.jl --no-push          # commit in a local worktree, print the push command
#
# Before deploying:
#   julia --project=docs docs/generate_plots.jl            # if plotting code changed
#   julia --project=docs docs/verify_examples.jl --all     # CI no longer verifies examples

const DOCS = @__DIR__
const ROOT = normpath(joinpath(DOCS, ".."))
const BUILD = joinpath(DOCS, "build")

git(args::Cmd; dir::AbstractString=ROOT) = run(Cmd(`git $args`; dir))

function parse_deploy_args(args)
    target, skip_build, do_push = "dev", false, true
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--version"
            i == length(args) && error("--version requires an argument, e.g. --version v0.6.0")
            target = args[i+1]
            occursin(r"^v\d+\.\d+\.\d+$", target) || error("version must look like v0.6.0, got $target")
            i += 1
        elseif a == "--skip-build"
            skip_build = true
        elseif a == "--no-push"
            do_push = false
        else
            error("unknown argument $a (expected --version vX.Y.Z, --skip-build, --no-push)")
        end
        i += 1
    end
    return target, skip_build, do_push
end

function build_docs()
    ENV["CI"] = "true"           # prettyurls + canonical links in make.jl
    ENV["DOCS_DRAFT"] = "false"  # never deploy a draft build (no @example output)
    @info "Building documentation — this executes every @setup/@example block"
    include(joinpath(DOCS, "make.jl"))
end

# One symlink per minor series pointing at its highest patch, stable -> newest,
# and versions.js regenerated from the folders actually present (idempotent).
function update_version_index(wt)
    vers = sort!([VersionNumber(d) for d in readdir(wt)
                  if occursin(r"^v\d+\.\d+\.\d+$", d) && isdir(joinpath(wt, d)) && !islink(joinpath(wt, d))])
    isempty(vers) && error("no release folders found on gh-pages")
    newest = last(vers)
    minors = Dict{Tuple{Int,Int},VersionNumber}()
    for v in vers
        k = (Int(v.major), Int(v.minor))
        minors[k] = max(get(minors, k, v), v)
    end
    links = Dict("stable" => "v$newest")
    for ((mj, mn), v) in minors
        links["v$mj.$mn"] = "v$v"
    end
    for (link, dst) in links
        p = joinpath(wt, link)
        rm(p; force=true)
        symlink(dst, p)
    end
    doc_versions = ["stable"; ["v$mj.$mn" for (mj, mn) in sort!(collect(keys(minors)); rev=true)]; "dev"]
    write(joinpath(wt, "versions.js"),
        "var DOC_VERSIONS = [\n" * join("  \"$v\",\n" for v in doc_versions) *
        "];\nvar DOCUMENTER_NEWEST = \"v$newest\";\nvar DOCUMENTER_STABLE = \"stable\";\n")
end

function deploy(target, do_push)
    isfile(joinpath(BUILD, "index.html")) ||
        error("docs/build/index.html not found — build first (drop --skip-build)")
    isfile(joinpath(BUILD, "data.html")) &&
        error("docs/build was built without prettyurls (CI=true) — re-run without --skip-build")

    git(`fetch origin gh-pages`)
    wt = joinpath(mktempdir(), "gh-pages")
    git(`worktree add --detach $wt origin/gh-pages`)
    kept = false
    try
        dest = joinpath(wt, target)
        rm(dest; recursive=true, force=true)
        cp(BUILD, dest)
        isdev = target == "dev"
        write(joinpath(dest, "siteinfo.js"),
            "var DOCUMENTER_CURRENT_VERSION = \"$target\";\nvar DOCUMENTER_IS_DEV_VERSION = $isdev;\n")
        isdev || update_version_index(wt)

        git(`add -A`; dir=wt)
        if success(Cmd(`git diff --cached --quiet`; dir=wt))
            @info "gh-pages already up to date — nothing to deploy"
            return
        end
        sha = readchomp(Cmd(`git rev-parse --short HEAD`; dir=ROOT))
        branch = readchomp(Cmd(`git rev-parse --abbrev-ref HEAD`; dir=ROOT))
        git(`commit -m "docs: deploy $target from $branch@$sha"`; dir=wt)
        if do_push
            git(`push origin HEAD:gh-pages`; dir=wt)
            @info "Deployed $target — https://api.friedman.jp/MacroEconometricModels.jl/$target/"
        else
            kept = true
            @info """
            Commit created but NOT pushed. To publish:
                git -C $wt push origin HEAD:gh-pages
                git -C $ROOT worktree remove --force $wt
            """
        end
    finally
        kept || git(`worktree remove --force $wt`)
    end
end

function main()
    target, skip_build, do_push = parse_deploy_args(ARGS)
    skip_build || build_docs()
    deploy(target, do_push)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
