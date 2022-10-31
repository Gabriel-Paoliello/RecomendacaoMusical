import requests

lista = ['27NovPIUIRrOZoCHxABJwK','7FdUvDkaE24o3FPIWTvzv2','06WUUNf7q18NZfjIsQFsfa','0pqnGHJpmpxLKifKRmU6WP','6QewNVIDKdSl8Y3ycuHIei']

stringIds = lista[0]  + '%2C' + lista[1] + '%2C' + lista[2] + '%2C' + lista[3] + '%2C' + lista[4]

url = "https://api.spotify.com/v1/recommendations?limit=2&market=ES&seed_tracks="+stringIds
print (url)

headers= {"Authorization: Bearer BQDr2bHEKl15Nvi3Cq0R4IZ_d5GzAL91x5jOehc-Lyji2Ik7kINISvcL5NV2dHvANI0Dkhzc9cHE_fR2klr9pONDHBViN4Whb88i5-VeHIKgrVU7MXxJPRea1C1whTIm7KGnOjMUc7mgcCIGsdeOBiF2Ey45yL9BkyMtvBScmKeF2Rqe"}

print (headers)

resp = requests.get(url,headers=headers)
print (resp.content)
