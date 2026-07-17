using PrettyTables
println("PT ", pkgversion(PrettyTables))
d = hcat(["r$i" for i in 1:21], collect(1:21))
io = IOBuffer(); pretty_table(io, d; backend=:text); s = String(take!(io))
println("default: r21? ", occursin("r21", s), " omitted? ", occursin("omitted", s))
io2 = IOBuffer(); pretty_table(IOContext(io2, :limit=>false), d; backend=:text); s2 = String(take!(io2))
println("nolimit-ctx: r21? ", occursin("r21", s2))
for kw in (:display_size, :fit_table_in_display_vertically, :vcrop_mode)
  try
    io3 = IOBuffer()
    if kw == :display_size; pretty_table(io3, d; backend=:text, display_size=(-1,-1))
    elseif kw == :fit_table_in_display_vertically; pretty_table(io3, d; backend=:text, fit_table_in_display_vertically=false)
    else; pretty_table(io3, d; backend=:text, vcrop_mode=:bottom); end
    s3=String(take!(io3)); println(kw, ": r21? ", occursin("r21", s3))
  catch e; println(kw, " ERR ", typeof(e)); end
end
