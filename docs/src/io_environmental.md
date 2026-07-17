# Environmental Extensions

Satellite (extension) accounts attach stressor flows — emissions, employment,
materials, water — to an [`IOData`](@ref) table and support consumption-based
accounting via the Leontief inverse.

```@example ioe
using MacroEconometricModels
io = load_example(:wiot)        # ships with "employment" and "CO2" accounts
nothing # hide
```

## Intensities and multipliers

For a stressor account with flows `F` (stressor × sector), the **intensities**
are ``S = F\hat{x}^{-1}`` and the consumption-based **emission multipliers** are
``M = S L``.

```@example ioe
intensities(io, "CO2")
```

```@example ioe
emission_multipliers(io, "CO2")
```

## Footprints

The **footprint** is the consumption-based account ``e = M Y + F_Y`` (stressor ×
final-demand category), with a per-sector contribution `by_sector`.

```@example ioe
fp = footprint(io, "CO2")
fp.total
```

## Adding an extension

`add_extension!` attaches a new stressor account and precomputes its intensities.

```@example ioe
add_extension!(io, "water", [12.0 28.0]; stressors=["H2O"], unit=["Ml"])
intensities(io, "water")
```

## API

```@docs
IOExtension
add_extension!
intensities
emission_multipliers
footprint
```
