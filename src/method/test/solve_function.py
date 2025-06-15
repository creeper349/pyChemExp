import sympy as sp

A,B,C=sp.symbols("A B C")
h1,h2=sp.symbols("h1 h2")

eq1=sp.Eq(A+B+C,0)
eq2=sp.Eq(-A*h1+B*h2,1)
eq3=sp.Eq(A*h1**2+B*h2**2,0)
sol=sp.solve((eq1,eq2,eq3),(A,B,C),simplify=True)
for name in ["A","B","C"]:
    print(f"{name} = {sp.simplify(sol[sp.Symbol(name)])}")