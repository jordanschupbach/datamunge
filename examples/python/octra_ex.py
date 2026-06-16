from pyoctra import octra

# dir(octra)

octra.hello()

dp = octra.DPair(1, 2)

s = octra.SVector(3)
s[0] = "1"
s[1] = "2"
s[2] = "3"
# s[3] = "4"  # <- Gets index out of bounds error


class TimesTwo(octra.Callback):
    def call(self, x):
        return x * 2.0


class TimesThree(octra.Callback):
    def call(self, x):
        return x * 3.0


# cb = TimesTwo()
cb = TimesThree()
print("call_with_callback(3.0) =", octra.call_with_callback(3.0, cb))
v = octra.make_dvector(1.0, 2.0, 3.0)
v2 = octra.map_dvector_with_callback(v, cb)
print("sum_dvector(map_dvector_with_callback(1,2,3)) =", octra.sum_dvector(v2))
