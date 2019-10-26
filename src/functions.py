# © by Josua Bürki, 2019
# In this file are all functions needed for the program
# This replaces a library with my own functions
import random


def random_numbers_gauss(mean, standard_deviation, numbers_of_rows, numbers_of_columns):
    result_matrix = []
    for i in range(numbers_of_rows):
        empty_matrix = []
        for j in range(numbers_of_columns):
            empty_matrix += [0]
        result_matrix += [empty_matrix]
        del empty_matrix
    for k in range(numbers_of_rows):
        for l in range(numbers_of_columns):
            result_matrix[k][l] = random.gauss(mean, standard_deviation)
    return result_matrix


def create_matrix(numbers_of_rows, numbers_of_columns, content):
    result_matrix = []
    for i in range(numbers_of_rows):
        empty_matrix = []
        for j in range(numbers_of_columns):
            empty_matrix += [content]
        result_matrix += [empty_matrix]
        del empty_matrix
    return result_matrix


def leaky_relu(argument):
    for i in range(len(argument)):
        for j in range(len(argument[0])):
            if argument[i][j] < 0:
                argument[i][j] = 0.01 * argument[i][j]
            elif argument[i][j] > 1:
                argument[i][j] = argument[i][j]
            else:
                argument[i][j] = argument[i][j]
    return argument


def leaky_relu_derivation(argument):
    for i in range(len(argument)):
        for j in range(len(argument[0])):
            if argument[i][j] < 0:
                argument[i][j] = 0.01
            elif argument[i][j] > 1:
                argument[i][j] = 1
            else:
                argument[i][j] = 1
    return argument


def matrix_transposition(matrix):
    result_matrix = []
    for i in range(len(matrix[0])):
        empty_matrix = []
        for j in range(len(matrix)):
            empty_matrix += [0]
        result_matrix += [empty_matrix]
        del empty_matrix
    for k in range(len(matrix)):
        for l in range(len(matrix[0])):
            result_matrix[l][k] = matrix[k][l]
    return result_matrix


def matrix_multiplication(first, second):
    if len(first[0]) == len(second):
        result_matrix = []
        for i in range(len(first)):
            empty_matrix = []
            for j in range(len(second[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(first)):
            for l in range(len(second[0])):
                for m in range(len(second)):
                    result_matrix[k][l] += first[k][m] * second[m][l]
        return result_matrix


def multiplication(multiplier, multiplicand):
    if isinstance(multiplier, list) and isinstance(multiplicand, list):
        if len(multiplier) == len(multiplicand) and len(multiplier[0]) == len(multiplicand[0]):
            result_matrix = []
            for i in range(len(multiplier)):
                empty_matrix = []
                for j in range(len(multiplier[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(multiplier)):
                for l in range(len(multiplier[0])):
                    result_matrix[k][l] = multiplier[k][l] * multiplicand[k][l]
            return result_matrix
        elif len(multiplier) == len(multiplicand) and len(multiplicand[0]) == 1:
            print("Matrix wrong dimension (multiplication)")
        elif len(multiplier[0]) == len(multiplicand[0]) and len(multiplicand) == 1:
            print("Matrix wrong dimension (multiplication)")
    elif isinstance(multiplier, float) and isinstance(multiplicand, list):
        result_matrix = []
        for i in range(len(multiplicand)):
            empty_matrix = []
            for j in range(len(multiplicand[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(multiplicand)):
            for l in range(len(multiplicand[0])):
                result_matrix[k][l] = multiplier * multiplicand[k][l]
        return result_matrix


def subtraction(minuend, subtrahend):
    if isinstance(minuend, list) and isinstance(subtrahend, list):
        if len(minuend) == len(subtrahend) and len(minuend[0]) == len(subtrahend[0]):
            result_matrix = []
            for i in range(len(minuend)):
                empty_matrix = []
                for j in range(len(minuend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(minuend)):
                for l in range(len(minuend[0])):
                    result_matrix[k][l] = minuend[k][l] - subtrahend[k][l]
            return result_matrix
        elif len(minuend) == len(subtrahend) and len(subtrahend[0]) == 1:
            print("Matrix wrong dimension (subtraction)")
        elif len(minuend[0]) == len(subtrahend[0]) and len(subtrahend) == 1:
            print("Matrix wrong dimension (subtraction)")


def addition(augend, addend):
    if isinstance(augend, list) and isinstance(addend, list):
        if len(augend) == len(addend) and len(augend[0]) == len(addend[0]):
            result_matrix = []
            for i in range(len(augend)):
                empty_matrix = []
                for j in range(len(augend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(augend)):
                for l in range(len(augend[0])):
                    result_matrix[k][l] = augend[k][l] + addend[k][l]
            return result_matrix
        elif len(augend) == len(addend) and len(addend[0]) == 1:
            print("Matrix wrong dimension (addition)")
        elif len(augend[0]) == len(addend[0]) and len(addend) == 1:
            print("Matrix wrong dimension (addition)")
    elif isinstance(augend, list) and isinstance(addend, float):
        result_matrix = []
        for i in range(len(augend)):
            empty_matrix = []
            for j in range(len(augend[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(augend)):
            for l in range(len(augend[0])):
                result_matrix[k][l] = augend[k][l] + addend
        return result_matrix


def copy_board(board_to_copy):
    # Replaces the copy module
    copied_board = []

    for i in range(len(board_to_copy)):
        copied_list = []
        for j in range(len(board_to_copy[0])):
            copied_list.append(board_to_copy[i][j])
        copied_board.append(copied_list)

    return copied_board


def board_to_number(board):
    number = 0
    for i in range(9):
        number += board[i] * 3 ** (8 - i)
    return number

