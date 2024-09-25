# 5장 1 2 3 4 6 7 9 10 11번 문제

import math
from tkinter import N
from turtle import st

'''5.5.1
‘racecar’, ‘토마토’, ‘stats’와 같이 앞뒤를 뒤집어도 똑같은 문자열을 회문palindrome이라고 한다. 문자열이 주어질 때, 그 문자열이 회문이면 Success를, 아니면 Fail를 출력하는 코드를 작성하여라.
'''

def is_palindrome(word):
    return 'Success' if word == word[::-1] else 'Fail'

test = input("Input: ")
print(f"Output: {is_palindrome(test)}")

'''5.5.2
거꾸로 노래 부르기 좋아하는 청개구리는 아래와 같이 노래를 부른다.

끼토산 야끼토 로디어 냐느가 총깡총깡 서면뛰 를디어 냐느가

문자열이 주어졌을 때, 공백을 단위로 각 단어를 거꾸로 출력하는 코드를 작성하여라.
단, 특정 노래의 가사는 줄바꿈 없이 주어진다.
'''

def reverse_song(song):
    return ' '.join([word[::-1] for word in song.split()])

test = "끼토산 야끼토 로디어 냐느가 총깡총깡 서면뛰 를디어 냐느가"
print(f"Output: {reverse_song(test)}")

'''5.5.3
카멜camel 표기법으로 작성된 문자열을 팟홀pothole 표기법(또는 스네이크snake 표기법)으로 변경하는 코드를 작성해보자.

카멜 표기법 : 소문자로 시작하고, 이어지는 단어의 시작은 대문자로 작성하는 표기법. 예를 들어, userName, printMessage, countA 등.

팟홀 표기법 또는 스네이크 표기법 : 모두 소문자를 사용하고, 단어 사이에 밑줄기호(_)를 사용하는 표기법. 예를 들어, user_name, print_message, count_a 등.
'''

def camel_to_snake(camel):
    snake = ''
    for char in camel:
        if char.isupper():
            snake += '_'
            snake += char.lower()
        else:
            snake += char
    return snake

test = input("Input: ")
print(f"Output: {camel_to_snake(test)}")

'''5.5.4
n! 팩토리얼factorial은 1 x 2 x ... x n 을 의미한다. 자연수 n이 주어졌을 때, n!의 끝에 있는 0 terminal zeros의 개수를 출력하는 코드를 작성하여라. n은 1보다 크고 100000보다 작은 자연수라고 가정하자.
'''
# def count_terminal_zeros(n):
#     i = 1
#     while(n//i > 0):
#         i *= 5
#     greatest_power = i//5
#     cnt= 0
#     while(greatest_power > 0):
#         cnt += n//greatest_power
#         greatest_power //= 5
#     return cnt

# test = int(input("Input: "))
# print(f"Output: {count_terminal_zeros(test)}")

def count_terminal_zeros(n):
    n_fac = math.factorial(n)
    cnt = 0
    while n_fac % 10 == 0:
        cnt += 1
        n_fac //= 10
    return cnt

'''5.5.6
어느 강좌의 수강생과 시험 점수가 주어졌을 때, 학점을 매기는 프로그램을 만들어보자.

수강생의 30% 이하가 A학점을 받을 수 있다.

수강생의 60% 이하가 A학점 또는 B학점을 받을 수 있다.

A학점과 B학점이 아닌 나머지는 C학점이다.

최대한 좋은 성적을 부여한다. 예를 들어, 10명 중 3등을 했다면, B, C등의 학점도 가능하지만 A학점을 받는다고 가정한다.

입력으로 학생들의 이름과 점수가 두 줄로 들어오는 데, 각 줄의 항목은 공백으로 구분된다.
'''

def get_grade(students):
    students = sorted(students, key=lambda x: x[1], reverse=True)
    n = len(students)
    grades = ['A']*int(n*0.3) + ['B']*int(n*0.3) + ['C']*(n-(2*int(n*0.3)))
    return {student[0]: grade for student, grade in zip(students, grades)}

names = input("Input: ").split()
grades = list(map(int, input().split()))
students = list(zip(names, grades))
print(f"Output: {get_grade(students)}")

'''5.5.7
독일 수학자인 콜라츠Collatz, L.는 1937년에 아래 알고리즘을 얼마나 많이 반복하면 최종적으로 숫자 1에 다다를 것인가를 질문했다.

주어진 숫자가 짝수라면 2로 나눈다.

주어진 숫자가 홀수라면 3을 곱한 다음 1을 더한다.

실제로 숫자 7부터 시작해서 위 과정을 16번 반복하면 1에 다다른다.
7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1

반면에 숫자 128부터 시작하면 7번만 반복하면 된다.
128, 64, 32, 16, 8, 4, 2, 1

콜라츠는 어떤 자연수로 시작하든지 반복작업이 언젠가는 끝난다고 추측했는데, 아직 언제 끝나는가는 수학적으로 알아내지 못했다. 이를 콜라츠 추측이라고 부른다.

자연수 n을 입력받아 위의 알고리즘을 몇 번 반복하면 1에 다다르는지를 아래와 같은 형식으로 반환하는 collatz() 함수를 정의하여라. 단, 100번이상 반복하는 경우 -1를 반환한다.
'''

def collatz(n):
    cnt = 0
    while n != 1:
        if cnt == 100:
            return -1
        n = n//2 if n%2 == 0 else 3*n+1
        cnt += 1
    return cnt

test = int(input("Input: "))
print(f"Output: {collatz(test)}")

'''5.5.9
A는 정수를 입력하는 일을 하고 있다. 이때 연속으로 같은 숫자를 입력하면 안되는데, 입력장치의 오류로 연속적으로 같은 숫자가 입력된 것을 발견하였다. A가 입력한 숫자 리스트를 인자로 받아 연속적으로 같은 숫자가 있다면 그 숫자는 하나만 남기고 나머지는 제거한 리스트를 반환하는 함수disconti_num()를 만들어라. 단, 입력의 순서는 유지되야 하고, 같은 숫자가 불연속적으로 여러 번 등장할 수 있다.
'''

def disconti_num(nums):
    return [nums[i] for i in range(len(nums)) if i == 0 or nums[i] != nums[i-1]]

test = list(map(int, input("Input: ").split()))
print(f"Output: {disconti_num(test)}")

'''5.5.10
여러 개의 다항식을 곱하는 프로그램을 만들어보자. 다항식은 각 항의 계수로 나타낼 수 있다. 예를 들어, 2는 
를, 1 3 4는 
를, 0 4 0 1은 
을 의미한다.

입력의 첫 줄에는 다항식의 개수 N(0<N<5)이 주어진다. 그 다음에 들어오는 N개의 줄에는 각 줄마다 다항식의 계수가 공백으로 분리되어 주어진다. 단, 입력으로 주어지는 각 다항식의 계수는 모두 정수이며, 최고차항의 계수는 0이 아니다.
'''

def poly_mul(polys):
    result = [1]
    for poly in polys:
        result = [sum([result[i-j]*poly[j] for j in range(len(poly)) if (i-j >= 0 and i-j <= len(result)-1)]) for i in range(len(result)+len(poly)-1)]
    return result

n = int(input("Input: "))
polys = [list(map(int, input().split())) for _ in range(n)]
print(f"Output: {' '.join(map(str, poly_mul(polys)))}")

'''5.5.11
주어진 빙고판에서 빙고의 총 개수를 세는 프로그램을 만들어보자. 빙고판의 행의 개수와 열의 개수는 모두 2이상이고, 행의 개수와 열의 개수는 같다. 빙고판은 o와 x로만 이루어져 있으며, 빙고 여부는 o 기준으로 생각한다. 입력의 첫 줄에는 빙고판 크기 N이 주어진다. 그 다음에 들어오는 N개의 줄에는 o와 x가 공백으로 분리되어 입력되는데, 각각 N개 넘게 입력하면 N개를 입력할 때까지 계속 입력을 받는다.
'''

def count_bingo(board):
    n = len(board)
    cnt = 0
    for i in range(n):
        if all([board[i][j] == 'o' for j in range(n)]):
            cnt += 1
        if all([board[j][i] == 'o' for j in range(n)]):
            cnt += 1
    if all([board[i][i] == 'o' for i in range(n)]):
        cnt += 1
    if all([board[i][n-i-1] == 'o' for i in range(n)]):
        cnt += 1
    return cnt

n = int(input("Input: "))
board = [input().split() for _ in range(n)]
print(f"Output: {count_bingo(board)}")