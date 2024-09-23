##################################
#                                #
# 1주차 실습 (9/7)                #
# 기초 I(기본 자료형, 변수, 함수)    #
# 13장 1 ~ 8번                    #
#                                #
##################################
"""
1.
다음을 연산기호를 이용하여 수식으로 표현하고, 그 결과를 확인하여라.
"""

# (1)
# 3 + 9 - 8

print('3 + 9 - 8 =', 3 + 9 - 8)
print('3 + 9 - 8 =', 3 + 9 - 8, sep='')
print(f'3 + 9 - 8 = {3 + 9 - 8}')
print('3 + 9 - 8 = {0}'.format(3 + 9 - 8))

# (2)
# (27 + 3) × 2 - 42 / 3

print("(27 + 3) × 2 - 42 / 3 = ", (27 + 3) * 2 - 42 / 3)

# (3)
# [{(9 / 2) × 12} - 7] × 3 - 1

print("[{(9 / 2) × 12} - 7] × 3 - 1 = ", (((9 / 2) * 12) - 7) * 3 - 1)

"""
2. 
문자열 Good morning! 을 출력하는 코드를 작성하여라.
"""

print("Good morning!")  # 큰 따옴표
print('Good morning!')  # 작은 따옴표

"""
3. 
Hello!를 3번 연속(Hello!Hello!Hello!)으로 출력하는 코드를 작성하여라.
"""

print('Hello!Hello!Hello!')  # 방법 (0)
print('Hello!' * 3)  # 방법 (1)

a = 'Hello!'
print(a * 3)  # 방법 (2)

"""
4.
사용자에게 문자열을 입력받고, 그 문자열을 5번 연속으로 출력하는 코드를 작성하여라.
예를 들어, 사용자가 Hello!를 입력했다면, Hello!Hello!Hello!Hello!Hello!를 출력한다.
"""

# b = input('문자열을 입력해주세요: ')
b = 'hi'
print(b * 5)

"""
5.
두 정수 a와 b를 입력받아 a + b, a - b, a * b, a / b, a // b, a % b를 출력하는 코드를 작성하여라.
"""

# (1) string 포매팅 (2) str으로 입력받은 값을 int로 형변환

# a = input('a = ')
a = 100
print(type(a))
a = int(a)
# b = int(input('b = '))
b = 3

print(f'{a} + {b} =', a + b)
print(f'{a} - {b} =', a - b)
print(f'{a} * {b} =', a * b)
print(f'{a} / {b} =', a / b)
print(f'{a} // {b} =', a // b)  # 몫
print(f'{a} % {b} =', a % b)  # 나머지

"""
6.
두 자연수 x와 y를 입력받아 두 수의 산술평균 (arithmetic mean), 기하평균 (geometric mean), 조화평균 (harmonic mean)을 출력하는 코드를 작성하여라.
"""

# (1) str -> int 형변환 (2) output의 자료형 이해

# x = int(input('x = '))
# y = int(input('y = '))
x = 3
y = 3
Ma = (x + y) / 2
Mg = (x * y)**0.5
Mh = 2 / (1 / x + 1 / y)  # 2xy / (x + y)
print('산술평균 =', Ma)
print('기하평균 =', Mg)
print('조화평균 =', Mh)

# 참고
print(type(Ma))
print(type(Mg))
print(type(Mh))

"""
7.
오늘이 무슨 요일인지 주어질 때, 100일 후는 무슨 요일인지를 출력하는 코드를 작성하여라.
단, 1234560은 각각 월화수목금토일을 의미한다. 즉, 2는 화요일이다.
Input : 2
Output : 4
(1) int로 형변환 (2) 간단한 알고리즘 사고: 100일 후 요일 계산 어떻게 할지?
"""
# today = int(input('오늘은 무슨 요일인가요?\n(월화수목금토일은 각각 1234560): '))
today = 2
result = (today + 100) % 7
print('100일 후는:', result)

today = '화'
weekday = {'월': 1, '화': 2, '수': 3, '목': 4, '금': 5, '토': 6, '일': 7}
result = (weekday[today] + 100) % 7
print('100일 후는:', result)

"""
8.
다음은 BTS의 butter 가사 일부분이다. 아래와 같이 문자열을 출력하는 코드를 작성하여라(따옴표 포함).

Smooth like "butter"
Like a criminal undercover
Gon' pop like trouble
Breakin' into your heart like that
"""
print(
    'Smooth like "butter"\nLike a criminal undercover\nGon\' pop like trouble\nBreakin\' into your heart like that'
)  # (1) 전체 따옴표 '' 사용

print(
    "Smooth like \"butter\"\nLike a criminal undercover\nGon' pop like trouble\nBreakin' into your heart like that"
)  # (2) 전체 따옴표 "" 사용

print("Smooth like \"butter\"")
print("Like a criminal undercover")
print("Gon' pop like trouble")
print("Breakin' into your heart like that")
# (3) \n 대신 print 함수 여러 번 사용

butter = """Smooth like \"butter\"
Like a criminal undercover
Gon\' pop like trouble
Breakin\' into your heart like that"""
print(butter)
# (4) 삼중 따옴표 사용

"""
10.
Key: width와 height을 변수로 지정. 그림 그려서 풀어본다
"""

width = 400
height = 300
w_tiles = width // 30
h_tiles = height // 30
whole_tiles = w_tiles * h_tiles
part_tiles = (w_tiles + (width % 30 != 0)) * (h_tiles +
                                              (height % 30 != 0)) - whole_tiles
print(whole_tiles, part_tiles)
# 참고: if문, for문 쓰면 더 간단해질 수 있음. -> 4주차에 배울 것

"""
11.
Key: 그림 그려보고 max 조건, min 조건 찾는다
"""

# N = int(input('의자 수를 입력하세요: '))
N = 10
max_chairs = (N + 1) // 2
min_chairs = (N + 2) // 3  # 그림 그려가면서 해볼 것
print(max_chairs, min_chairs)