import sys

def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

try:
    with open('primes100.py', 'w') as file:
        file.write("# List of prime numbers up to 100\n")
        for number in range(2, 101):
            if is_prime(number):
                file.write(f"{number}\n")
except IOError as e:
    print(f"Error writing to file: {e}", file=sys.stderr)