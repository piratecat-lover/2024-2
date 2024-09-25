# To print single quotes in a triple-quoted string, use backslash + single quote.
# print('''\'fat
#     cat\'''')

# To print brackets in a f-string, use double brackets
# a = 30
# print(f"{{a}}") # {a}

def poly_mul(polys):
    result = [1]
    for poly in polys:
        result = [sum([result[i-j]*poly[j] for j in range(len(poly)) if (i-j >= 0 and i-j <= len(result)-1)]) for i in range(len(result)+len(poly)-1)]
    return result

n = int(input("Input: "))
polys = [list(map(int, input().split())) for _ in range(n)]
print(f"Output: {' '.join(map(str, poly_mul(polys)))}")