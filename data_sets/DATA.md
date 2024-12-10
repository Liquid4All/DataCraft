# 1. Language (Plain Text, e.g., Captions)

```json
{
  "text": "Your caption or text here."
}
```

# 2. Conversation (Multi-Modal)

```json
{
  "conversations": [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "value": "You are a helpful AI assistant."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "value": "Can you describe this image?"
          },
          {
            "type": "image",
            "value": "/path/to/file"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "value": "The image shows a beautiful sunset over the mountains."
          }
        ]
      },
      // Additional Messages
    ]
}
```
- Roles should be in "system", "user", or "assistant"
- If there are multiple (seperate) conversations per image, they should be stored as seperate lines. E.g. return a list longer than the batch size. 

# 3. Bounding Box (Image Specific)

```json
{
  "bbox": [
    {
      "label": "object_label",
      "coordinates": [x1, y1, x2, y2]
    }
  ]
}
```
Ensure that the coordinates represent (0,0) as the top-left corner of the image, and ideally are normalized 0-1.

# 4. Points
```json

{

  "points": [

    {

      "label": "point_label",

      "coordinates": [x, y]

    }

  ]

}

```
Ensure that the coordinates represent (0,0) as the top-left corner of the image, and ideally are normalized 0-1.

# 5. Tabular Data

```json
{
  "tables": [
    [
      {
        "column1": "value1",
        "column2": "value2",
        // additional columns as needed
      },
      {
        "column1": "value3",
        "column2": "value4",
        // additional columns as needed
      }
      // additional rows as needed
    ],
    // additional tables as needed
  ]
}
```