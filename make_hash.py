import bcrypt

password = "Demo@2026!"   # ← change if you want

hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

print("\nCopy this hash into config.yaml:\n")
print(hashed)