# Utilities for selectively enabling JIT compilation.


from beartype import beartype
from beartype.typing import Callable
from jaxtyping import jaxtyped
import os


if os.getenv("NONJIT") == "1":  # pragma: no cover

    print("*** NOTE: `NONJIT` enabled")

    def jit(*static_argnums) -> Callable[[Callable], Callable]:
        return jaxtyped(typechecker=beartype)  # itself a function

else:  # pragma: no cover

    print("*** NOTE: `NONJIT` NOT enabled; JIT-compiling everything...")

    from jax import jit as jax_jit
    from jax.experimental.checkify import checkify, all_checks

    def jit(*static_argnums) -> Callable[[Callable], Callable]:
        def partially_applied(f: Callable) -> Callable:
            def checkified(*args, **kwargs):
                y = checkify(
                    jaxtyped(f, typechecker=beartype),
                    errors=all_checks,
                )(*args, **kwargs)
                print(f"Compiling {getattr(f, '__qualname__', 'an unnamed function')}...")
                return y

            def handle_err(*args, **kwargs):
                err, y = jax_jit(
                    checkified,
                    static_argnums=static_argnums,
                )(*args, **kwargs)
                err.throw()
                return y

            return handle_err

        return partially_applied
