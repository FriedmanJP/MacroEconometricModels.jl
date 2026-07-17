# One-time script to convert Acemoglu et al. (2019) DDCG CSV to TOML
# Run: julia scripts/convert_ddcg.jl
# Do NOT commit this file.

using TOML

csv_path = "/Users/chung/Downloads/dgjt_replication_package/processed/lpdid_ddcg.csv"

# Read CSV bytes and sanitize non-ASCII (latin-1 encoded)
raw_bytes = read(csv_path)
raw = String(map(b -> b > 0x7f ? UInt8('.') : b, raw_bytes))

# Simple CSV parser handling quoted fields with commas
function parse_csv_line(line::AbstractString)
    fields = String[]
    i = 1
    n = length(line)
    while i <= n
        if line[i] == '"'
            # Quoted field — find closing quote
            j = i + 1
            while j <= n && line[j] != '"'
                j += 1
            end
            push!(fields, String(line[i+1:j-1]))
            # Skip closing quote and comma
            i = j + 2
        else
            j = i
            while j <= n && line[j] != ','
                j += 1
            end
            push!(fields, String(line[i:j-1]))
            i = j + 1
        end
    end
    fields
end

lines = split(raw, '\n')
header = parse_csv_line(lines[1])

# Find column indices
wbcode_idx = findfirst(==("wbcode"), header)
year_idx = findfirst(==("year"), header)
y_idx = findfirst(==("y"), header)
dem_idx = findfirst(==("dem"), header)

println("Columns: wbcode=$wbcode_idx, year=$year_idx, y=$y_idx, dem=$dem_idx")

countries = String[]
years = Int[]
y_vals = Float64[]
dem_vals = Float64[]

for i in 2:length(lines)
    line = strip(lines[i])
    isempty(line) && continue
    fields = parse_csv_line(line)
    length(fields) >= dem_idx || continue

    push!(countries, strip(fields[wbcode_idx]))
    push!(years, parse(Int, strip(fields[year_idx])))
    yv = strip(fields[y_idx])
    push!(y_vals, isempty(yv) ? NaN : parse(Float64, yv))
    dv = strip(fields[dem_idx])
    push!(dem_vals, isempty(dv) ? NaN : parse(Float64, dv))
end

# Build unique country list (preserving order)
seen = Set{String}()
unique_countries = String[]
for c in countries
    if !(c in seen)
        push!(unique_countries, c)
        push!(seen, c)
    end
end

unique_years = sort(unique(years))

println("Rows: $(length(y_vals))")
println("Countries: $(length(unique_countries))")
println("Years: $(unique_years[1])-$(unique_years[end]) ($(length(unique_years)))")
println("Non-missing y: $(count(!isnan, y_vals))")
println("Democracy=1: $(count(==(1.0), dem_vals))")

# Build TOML dict
d = Dict{String,Any}(
    "metadata" => Dict{String,Any}(
        "n_countries" => length(unique_countries),
        "n_years" => length(unique_years),
        "years" => unique_years,
        "frequency" => "Yearly",
        "desc" => "Acemoglu, Naidu, Restrepo & Robinson (2019) Democracy-GDP Panel",
        "source_refs" => ["Acemoglu2019_DDCG", "DubeGirardiJordaTaylor2025"],
    ),
    "variables" => Dict{String,Any}(
        "names" => ["y", "dem"],
    ),
    "countries" => Dict{String,Any}(
        "codes" => unique_countries,
        "names" => unique_countries,
    ),
    "descriptions" => Dict{String,Any}(
        "y" => "Log GDP per capita",
        "dem" => "Democracy indicator (binary, 0/1)",
    ),
    "data" => Dict{String,Any}(
        "y" => y_vals,
        "dem" => dem_vals,
    ),
)

toml_path = joinpath(@__DIR__, "..", "data", "ddcg.toml")
open(toml_path, "w") do io
    TOML.print(io, d)
end

println("\nWrote $toml_path")
println("File size: $(round(filesize(toml_path) / 1024; digits=1)) KB")
