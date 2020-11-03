using CSV
using DataFrames

raw=DataFrame(CSV.File("notebook/data-aa-58-200.tsv", delim="\t", ))

concepts = unique(raw[!,:CONCEPT])
langs = unique(raw[!,:ISO_CODE])

outtputArr = Array{String, 2}(undef, length(langs), length(concepts))

for (ind, concept) in enumerate(concepts)
    for_conc = filter(row -> row.CONCEPT == concept, raw)

    for (ind2, lang) in enumerate(langs)
        cc = filter(row -> row.ISO_CODE == lang, for_conc).COGID

        if length(cc) == 0
            cc1 = "?"

        else
                cc1 = string(cc[1])

        end
        outtputArr[ind2, ind] = cc1
    end
end
real_out = similar(outtputArr, Float64)
for i in 1:size(outtputArr, 2)
    vals = sort(unique(outtputArr[:,i]))
    for j in 1:size(outtputArr, 1)
        if outtputArr[j,i] != "?"
            real_out[j, i] = findfirst(v -> v == outtputArr[j,i], vals)
        else
            real_out[j, i] = 0
        end
    end
end
