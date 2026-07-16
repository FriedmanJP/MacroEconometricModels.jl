using MacroEconometricModels
ap = MacroEconometricModels.adf_pvalue
println("c -3.4335 => ", ap(-3.4335,:constant,200,0), " (want 0.0100)")
println("c -2.8621 => ", ap(-2.8621,:constant,200,0), " (want 0.0500)")
println("c -2.5671 => ", ap(-2.5671,:constant,200,0), " (want 0.0999)")
println("c -2.0    => ", ap(-2.0,:constant,200,0), " (want 0.2866)")
println("c -1.61   => ", ap(-1.61,:constant,200,0), " (want 0.478)")
for (cvs,reg) in (((-2.5658,-1.9393,-1.6156),:none),((-3.9638,-3.4126,-3.1279),:trend))
  println(reg," ",[round(ap(c,reg,200,0),digits=4) for c in cvs], " (want ~0.01,0.05,0.10)")
end
println("c 0.0 => ", ap(0.0,:constant,200,0))
println("c 1.0 => ", ap(1.0,:constant,200,0))
println("c -1.94 => ", ap(-1.94,:constant,200,0), " old=0.1236")
grid=collect(-6.0:0.25:2.0); pv=[ap(t,:constant,200,0) for t in grid]
println("strictly decreasing? ", all(diff(pv).<0))
d=diff(pv); bad=findall(d.>=0); println("non-decreasing at idx: ", [(grid[i],pv[i],pv[i+1]) for i in bad])
