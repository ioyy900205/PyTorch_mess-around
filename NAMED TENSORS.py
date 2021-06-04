import torch
imgs = torch.randn(3, 1, 1, 2)
named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
print(named_imgs.names)

# Refine the last two dims to 'H' and 'W'. In Python 2, use the string '...'
# instead of ...
named_imgs = imgs.refine_names(..., 'H', 'W')
print(named_imgs.names)


def catch_error(fn):
    try:
        fn()
        assert False
    except RuntimeError as err:
        err = str(err)
        if len(err) > 180:
            err = err[:180] + "..."
        print(err)


named_imgs = imgs.refine_names('N', 'C', 'H', 'W')

# Tried to refine an existing name to a different name
catch_error(lambda: named_imgs.refine_names('N', 'C', 'H', 'width'))