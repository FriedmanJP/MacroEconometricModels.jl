module MacroEconometricModelsJLD2Ext

# JLD2 backend for versioned model serialization (T248 / #347).
# Overrides the more-generic src stubs `_write_model_container` /
# `_read_model_container`; loaded automatically when the user runs `using JLD2`.
# The container is a plain Dict{String,Any} of primitives/arrays/nested-dicts —
# JLD2 stores that robustly, which is the whole point of the versioned schema.

using MacroEconometricModels, JLD2
import MacroEconometricModels: _write_model_container, _read_model_container, SerializationError

function _write_model_container(path::AbstractString, container)
    JLD2.jldopen(path, "w") do f
        f["container"] = container
    end
    return path
end

function _read_model_container(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        haskey(f, "container") || throw(SerializationError(
            "file '$path' is not a MacroEconometricModels model file (missing 'container' group)"))
        f["container"]
    end
end

end # module
