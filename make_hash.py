import streamlit_authenticator as stauth

# Create hasher
hasher = stauth.Hasher()

# Hash ONE password at a time (your version requires single string)
admin_hash = hasher.hash("admin123")
basil_hash = hasher.hash("basil123")

print("Admin hash:", admin_hash)
print("Basil hash:", basil_hash)