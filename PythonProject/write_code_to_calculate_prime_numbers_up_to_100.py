def generate_primes(limit):
    primes = []
    for possiblePrime in range(2, limit + 1):
        isPrime = True
        for num in range(2, int(possiblePrime ** 0.5) + 1):
            if possiblePrime % num == 0:
                isPrime = False
                break
        if isPrime:
            primes.append(possiblePrime)
    return primes

# Example usage
primes_up_to_100 = generate_primes(100)
print(primes_up_to_100)