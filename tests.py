import torch
from EKFAC import EKFAC

def test_factorization(model, opt, x, y):
    criterion = torch.nn.MSELoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    opt.step()

    factorized = opt.approximate_Fisher_matrix(to_return=True)[0]
    actual = opt.compute_empirical_Fisher_matrix(to_return=True)[0]

    print(factorized)
    print(actual)


if __name__ == '__main__':
    n_in = 5
    n_out = 1
    batch_size = 100

    x = torch.rand((batch_size, n_in))
    y = torch.rand((batch_size, n_out))

    model = torch.nn.Linear(n_in, n_out, bias=False)
    opt = EKFAC(model)

    test_factorization(model, opt, x, y)
