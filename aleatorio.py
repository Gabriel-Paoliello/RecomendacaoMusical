import requests

lista = ['27NovPIUIRrOZoCHxABJwK','7FdUvDkaE24o3FPIWTvzv2','06WUUNf7q18NZfjIsQFsfa','0pqnGHJpmpxLKifKRmU6WP','6QewNVIDKdSl8Y3ycuHIei']

stringIds = lista[0]  + '%2C' + lista[1] + '%2C' + lista[2] + '%2C' + lista[3] + '%2C' + lista[4]

url = "https://api.spotify.com/v1/recommendations?limit=2&market=ES&seed_tracks="+stringIds
print (url)

headers= {"Authorization": "Bearer BQAPx3bQ1wpYEDz4ZJTIku7yzOTXfIbosJpfXQN9WsJBKmRLWJIFYBDuwThIM2rijVG9igWXTrfcIaTA2scRgGwOctjhLVLYVDtMaOhMTRYTBIBzXiOlAgDo3TAm5D6-jmgjhmVl3AOix8WASQJeOL0T7erw5Okxb0ePq6HgQbkYFyRR"}

print (headers)

resp = requests.get(url,headers=headers)
print (resp.content)