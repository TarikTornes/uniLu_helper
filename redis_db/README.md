#Build
```
docker build -t redis-db .
```

# Run
```
docker run --name redis-server -p 6379:6379 -d redis-db
```


