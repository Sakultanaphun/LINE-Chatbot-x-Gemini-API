const { GoogleGenerativeAI } = require("@google/generative-ai");

const apiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
  model: "gemini-1.5-flash-8b",
  systemInstruction: "คุณคือผู้เชี่ยวชาญด้านกฎหมาย การเมือง และการบริหารราชการไทย ที่ได้รับการรับรอง โดยเฉพาะอย่างยิ่งในระบบรัฐสภาและการบริหารพรรคการเมือง คุณมีความรู้เชิงลึกเกี่ยวกับการทำสัญญา การร่างหนังสือ และการจัดทำแบบฟอร์มต่าง ๆ คุณสามารถวิเคราะห์สถานการณ์ทางการเมืองอย่างเป็นกลาง พร้อมให้คำแนะนำที่สอดคล้องกับหลักธรรมาภิบาลและกฎหมาย คุณมีข้อมูลเกี่ยวกับนายอโนชา สกุลธนพันธ์ (โค้ชพรรคใหม่) และสามารถเสนอแนะแนวทางการทำงานที่เหมาะสมเพื่อพัฒนาสังคมและแก้ไขปัญหาความเดือดร้อนของประชาชน",
});

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 40,
  maxOutputTokens: 8192,
  responseMimeType: "text/plain",
};

const textOnly = async prompt => {
  const result = await model.generateContent(prompt);
  return result.response.text();
};

const multimodal = async imageBinary => {
  const visionModel = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
  const prompt = "ช่วยบรรยายภาพนี้ให้หน่อย";
  const mimeType = "image/png";

  const imageParts = [
    {
      inlineData: {
        data: Buffer.from(imageBinary, "binary").toString("base64"),
        mimeType
      }
    }
  ];

  const result = await visionModel.generateContent([prompt, ...imageParts]);
  return result.response.text();
};

const chat = async prompt => {
  const chatSession = model.startChat({
    generationConfig,
    history: [
      {
        role: "user",
        parts: [{ text: "สวัสดีครับ" }],
      },
      {
        role: "model",
        parts: [{ text: "สวัสดีครับ" }],
      }
    ]
  });

  const result = await chatSession.sendMessage(prompt);
  return result.response.text();
};

module.exports = { textOnly, multimodal, chat };