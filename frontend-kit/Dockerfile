FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./ 

RUN rm -f package-lock.json && \
    npm install --no-cache-dir --arch=x64 --platform=linux --libc=musl

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"] 
