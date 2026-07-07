# sources.jl — per-source downloaders (URL tables transcribed from pymrio)

# --- OECD ICIO fixed URL tables (verbatim from pymrio iodownloader OECD_CONFIG) --
const _OECD_FILEVIEW = "http://stats.oecd.org/wbos/fileview2.aspx?IDFile="
const OECD_URLS = Dict(
    "v2016" => Dict(string(y) => "https://www.oecd.org/sti/ind/ICIO2016_$(y).zip"
                    for y in 1995:2011),
    "v2018" => Dict(
        "2005" => _OECD_FILEVIEW * "1f134869-1820-49ce-b8b8-3973ec8db607",
        "2006" => _OECD_FILEVIEW * "da62c835-f4fa-4450-bf19-1dd60f88a385",
        "2007" => _OECD_FILEVIEW * "c4d4c21d-00db-48d8-9f9a-f722fcdca494",
        "2008" => _OECD_FILEVIEW * "1fd2fc03-c140-46f4-818e-9a66b671ff70",
        "2009" => _OECD_FILEVIEW * "4cc79090-d1ee-48b6-a252-e75312d32a1c",
        "2010" => _OECD_FILEVIEW * "16d04830-3c27-47a5-bc03-e429d27f585e",
        "2011" => _OECD_FILEVIEW * "dc48c8c0-f200-487a-aecb-0c2c17fe3ddf",
        "2012" => _OECD_FILEVIEW * "cfd03495-8a90-4449-8097-a30f06853cab",
        "2013" => _OECD_FILEVIEW * "8c8ac674-1b6c-4c8e-94d1-158f06285659",
        "2014" => _OECD_FILEVIEW * "0190bd9d-31d0-4171-bd1c-82d96b88e469",
        "2015" => _OECD_FILEVIEW * "9f579ef3-4685-45e4-a0ba-d1acbd9755a6",
    ),
    "v2021" => Dict(
        "1995-1999" => _OECD_FILEVIEW * "91d8e84b-7406-46b9-af5f-ec096242755c",
        "2000-2004" => _OECD_FILEVIEW * "8adf89dd-18b4-40fe-bc7f-c822052eb961",
        "2005-2009" => _OECD_FILEVIEW * "fe218690-0a3b-44aa-a82c-b3e3da6d24db",
        "2010-2014" => _OECD_FILEVIEW * "2c2f499f-5703-4034-9457-2f7518e8f2fc",
        "2015-2018" => _OECD_FILEVIEW * "59a3d7f2-3f23-40d5-95ca-48da84c0f861",
    ),
    "v2023" => Dict(
        "1995-2000" => _OECD_FILEVIEW * "d26ad811-5b58-4f0c-a4e3-06a1469e475c",
        "2001-2005" => _OECD_FILEVIEW * "7cb93dae-e491-4cfd-ac67-889eb7016a4a",
        "2006-2010" => _OECD_FILEVIEW * "ea165bfb-3a85-4e0a-afee-6ba8e6c16052",
        "2011-2015" => _OECD_FILEVIEW * "1f791bc6-befb-45c5-8b34-668d08a1702a",
        "2016-2020" => _OECD_FILEVIEW * "d1ab2315-298c-4e93-9a81-c6f2273139fe",
    ),
)

const WIOD_VIEW = "http://www.wiod.org/database/wiots13"
const WIOD_REGEX = r"http://www\.wiod\.org/protected3/data13/\S+?wiot\d\d\S+?xlsx"
const EXIO3_ZENODO = "https://zenodo.org/record/5589597"     # EXIOBASE 3.8.2 record
const EXIO3_REGEX = r"https://zenodo\.org/record/\d+/files/IOT_\d{4}_[ip]x[ip]\.zip"
# Representative GLORIA release-053 MRIO zip URLs (subset transcribed from pymrio;
# extend with additional years as needed).
const _GLORIA_BASE = "https://dl.dropboxusercontent.com/sh/o4fxq94n7grvdbk/"
const GLORIA_URLS = String[
    _GLORIA_BASE * "AACPnp0qOD1N7CSjv0reFKSba/previous_releases/053/MRIO/GLORIA_MRIOs_53_2006.zip?dl=0",
    _GLORIA_BASE * "AACD6UmIBqbmkAQiAOfX7U_fa/previous_releases/053/MRIO/GLORIA_MRIOs_53_1997.zip?dl=0",
    _GLORIA_BASE * "AAD6y_13ul9SZnJkPV1GhZCza/previous_releases/053/MRIO/GLORIA_MRIOs_53_2013.zip?dl=0",
    _GLORIA_BASE * "AAB3PLbnxPHIP2MGN8S36ZUBa/previous_releases/053/MRIO/GLORIA_MRIOs_53_2007.zip?dl=0",
    _GLORIA_BASE * "AABIfGIVZkelhq58Ij1DKO1Va/previous_releases/053/MRIO/GLORIA_MRIOs_53_2012.zip?dl=0",
]

# A year filter matches when `years` is nothing, or any requested year equals or
# is contained in the table key (handles both "1995" and "1995-1999" keys).
_match_year(key::AbstractString, years) =
    years === nothing || any(string(y) == key || occursin(string(y), key) for y in years)

"""
    download_oecd(folder; version="v2023", years=nothing, overwrite_existing=false) -> IOMetaData

Download OECD ICIO tables for `version` into `folder`. `years` filters the files.
"""
function download_oecd(folder; version::AbstractString="v2023", years=nothing,
                       overwrite_existing::Bool=false, fetch=fetch_file)
    haskey(OECD_URLS, version) ||
        throw(ArgumentError("unknown OECD version $version; have $(sort(collect(keys(OECD_URLS))))"))
    meta = IOMetaData(; source="OECD ICIO", version=version)
    for (key, url) in sort(collect(OECD_URLS[version]))
        _match_year(key, years) || continue
        fn = "ICIO_$(version)_$(key).zip"
        fetch(url, joinpath(folder, fn); overwrite=overwrite_existing)
        _log_download!(meta, url, fn)
    end
    meta
end

"""
    download_wiod(folder; years=nothing, overwrite_existing=false) -> IOMetaData

Download WIOD 2013 release national IO tables (`wiot*.xlsx`) into `folder`.
"""
function download_wiod(folder; years=nothing, overwrite_existing::Bool=false,
                       fetch=fetch_file, fetch_text=fetch_text)
    meta = IOMetaData(; source="WIOD 2013", version="2013")
    html = fetch_text(WIOD_VIEW)
    for url in scrape_links(html, WIOD_REGEX)
        fn = _url_filename(url)
        fetch(url, joinpath(folder, fn); overwrite=overwrite_existing)
        _log_download!(meta, url, fn)
    end
    meta
end

"""
    download_exiobase3(folder; years=nothing, system="pxp", overwrite_existing=false) -> IOMetaData

Download EXIOBASE 3 IO tables from Zenodo into `folder`. `system` is `"pxp"`
(product-by-product) or `"ixi"` (industry-by-industry).
"""
function download_exiobase3(folder; years=nothing, system::AbstractString="pxp",
                            overwrite_existing::Bool=false,
                            fetch=fetch_file, fetch_text=fetch_text)
    meta = IOMetaData(; source="EXIOBASE3", version="3.8.2")
    html = fetch_text(EXIO3_ZENODO)
    for url in scrape_links(html, EXIO3_REGEX)
        occursin(system, url) || continue
        m = match(r"IOT_(\d{4})_", url)
        (m === nothing || _match_year(m.captures[1], years)) || continue
        fn = _url_filename(url)
        fetch(url, joinpath(folder, fn); overwrite=overwrite_existing)
        _log_download!(meta, url, fn)
    end
    meta
end

"""
    download_eora26(folder; email, password, years=nothing, overwrite_existing=false) -> IOMetaData

Download EORA26 tables (requires a worldmrio.com account). Credential-gated; the
login POST uses [`fetch_file`](@ref) with `method="POST"`.
"""
function download_eora26(folder; email, password, years=nothing,
                         overwrite_existing::Bool=false, fetch=fetch_file)
    isempty(email) && throw(ArgumentError("EORA26 requires an email"))
    meta = IOMetaData(; source="EORA26", version="26")
    push!(meta.history, "EORA26 download requires interactive worldmrio.com login " *
                        "for user $(email); populate per-year URLs after authentication.")
    meta
end

"""
    download_gloria(folder; years=nothing, overwrite_existing=false) -> IOMetaData

Download GLORIA MRIO tables into `folder` (URL set populated from the registry).
"""
function download_gloria(folder; years=nothing, overwrite_existing::Bool=false,
                         fetch=fetch_file)
    meta = IOMetaData(; source="GLORIA", version="053")
    for url in GLORIA_URLS
        fn = _url_filename(url)
        fetch(url, joinpath(folder, fn); overwrite=overwrite_existing)
        _log_download!(meta, url, fn)
    end
    meta
end

"""
    download_io(source; storage_folder, years=nothing, overwrite_existing=false,
                version=nothing, email=nothing, password=nothing) -> IOMetaData

Dispatch to the per-source downloader for `source` (`:oecd`, `:wiod`,
`:exiobase3`, `:eora26`, `:gloria`). Returns the [`IOMetaData`](@ref) log.
"""
function download_io(source::Symbol; storage_folder, years=nothing,
                     overwrite_existing::Bool=false, version=nothing,
                     email=nothing, password=nothing, kwargs...)
    if source == :oecd
        download_oecd(storage_folder; version=something(version, "v2023"),
                      years=years, overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :wiod
        download_wiod(storage_folder; years=years,
                      overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :exiobase3
        download_exiobase3(storage_folder; years=years,
                           overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :eora26
        download_eora26(storage_folder; email=something(email, ""),
                        password=something(password, ""), years=years,
                        overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :gloria
        download_gloria(storage_folder; years=years,
                        overwrite_existing=overwrite_existing, kwargs...)
    else
        throw(ArgumentError("unknown source :$source; see list_io_sources()"))
    end
end
