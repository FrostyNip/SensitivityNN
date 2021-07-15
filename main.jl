## Multivariable Function Approximator For Testing
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux
using Random
using Plots


function custom_train(datapoints, noise_scalar)
    X = rand(datapoints, 2)
    Y=[]
    for i in 1:datapoints
        push!(Y, exp(X[i,1]+X[i,2] + noise_scalar * randn(1)[1]))
    end

    data=[];
    for i in 1:datapoints
        push!(data, (X[i,1:2], Y[i]))
    end

    model = Chain(
        Dense(2,10),
        Dense(10,10,σ),
        Dense(10,1)
    )

    loss(x, y) = Flux.mse(model(x), y)
    para = Flux.params(model)
    opt = ADAM()

    epochs=1000
    for i in 1:epochs
        trainset = Random.shuffle(data)[1:32]
        Flux.train!(loss, para, trainset, opt)
    end


    return model
end



test1 = collect(range(0,1,length=21))
test2 = collect(range(0,1,length=21))
exact=[];dim1=[];dim2=[];
for i in 1:length(test1)
    for j in 1:length(test2)
        push!(exact, exp(test1[i] + test2[j]))
        push!(dim1, test1[i])
        push!(dim2, test2[j])
    end
end

datalength = [100,1000]
noise_std =  [0.0,0.01]
err_matrix = zeros(length(datalength),length(noise_std))
for i in 1:length(datalength)
    for j in 1:length(noise_std)

        model = custom_train(datalength[i],noise_std[j])

        plotlist=[]
        N = length(dim1)
        for itr in 1:N
            push!(plotlist, model([dim1[itr],dim2[itr]])[1])
        end

        global mse = 0.0
        for itr in 1:N
            ŷ = model([dim1[itr],dim2[itr]])[1]
            global mse += 1/N * (ŷ - exact[itr]) ^ 2
        end

        datapoints = datalength[i]
        noise_scalar = j
        scatter(dim1,dim2,plotlist,title="Points = $datapoints, σ = $noise_scalar",
                label="Predicted")
        scatter!(dim1,dim2,exact,label="Exact")
        savefig("plot_N_$datapoints std_$noise_scalar")

        global err_matrix[i,j] = mse
    end
end
