# Enhanced RAG Pipeline Evaluation Report

## Overview

This report evaluates an enhanced RAG pipeline with the following components:

1. **Retrieval**: MRL embeddings (256d) for document retrieval
2. **Reasoning**: DeepSeek-R1 (deepseek-r1-distill-llama-70b) for legal reasoning
3. **Answer Generation**: Mistral-7B for final answer generation

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Reasoning Tokens | 14677 |
| Total Answer Tokens | 18325 |
| Avg. Reasoning Time | 3.73s |

## Test Queries

1. What are the requirements for temporary buildings?
2. What are the requirements for advertising boards and hoardings?
3. What are the specifications for communication towers?
4. What are the requirements for changing the use of a building?
5. What provisions are made for qualified persons to certify building works?
6. What happens if the services of a Qualified Person are terminated during construction?
7. What are the requirements for developments near expressways?
8. What is the requirement for display notices during construction?
9. How long is a Development Plan valid?
10. I need to erect a temporary shed for a religious festival that will last for 14 days. What specific approval process applies, and what documentation must I submit?

## Query: "What are the requirements for temporary buildings?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the requirements for temporary buildings based on the provided context. Let me go through each document step by step.

Starting with Document 1, it mentions that a Temporary Building can be granted by the Relevant Authority with or without conditions and is valid for up to one year. It also lists what a Temporary Building is, like sheds for entertainment, funeral, religious purposes, builder sheds, containers, etc. So, the first requirement seems to be obtaining permission from the Relevant Authority, which can include conditions, and the building can't stay longer than a year unless specified.

Moving to Document 2, it talks about the Relevant Authority permitting Temporary Buildings to facilitate building works for a particular period as per Regulation 106. It also mentions that insurance is needed if the building involves pilling or ground excavation. So, another requirement is insurance in certain cases, and there's a specific regulation (106) that applies here.

Document 3 defines a Temporary Building as one permitted for a specified period, after which it must be demolished. It also mentions terms like Terrace House and Urban Development Area, but those might not be directly relevant here. So, the key point is that the building is temporary and must be removed after the period ends.

Document 4 seems similar to Document 1 and 2. It reiterates Regulation 106, stating that a plan or sketch with a written request must be submitted to the Relevant Authority. Permission can have conditions, and the building must comply with other regulations. So, submitting plans and a written request is another requirement.

Document 5 discusses various building services and designs that must comply with regulations and standards. It mentions that if necessary, the Relevant Authority might require detailed plans. So, compliance with design and safety standards is another requirement, especially if the Authority deems it necessary.

Putting it all together, the requirements for temporary buildings are:

1. Obtain permission from the Relevant Authority, which may include conditions.
2. The permission is valid for up to one year, as specified.
3. Submit a plan or sketch along with a written request.
4. Ensure the building is used for specific temporary purposes as defined.
5. If the building involves pilling or excavation, obtain insurance.
6. Comply with all relevant building regulations and standards.
7. The building must be demolished after the permitted period ends.

I should also note that some areas might need more information, like whether the building involves pilling or excavation, and ensuring that all submitted plans meet the necessary standards. If the Authority requires more detailed designs, those must be provided as well.
</think>

The requirements for temporary buildings, based on the provided context, are as follows:

1. **Obtain Permission**: Temporary buildings must be permitted by the Relevant Authority, which may include specific conditions.

2. **Validity Period**: The permission is granted for a specified period, not exceeding one year.

3. **Submission of Plans**: A plan or sketch along with a written request must be submitted to the Relevant Authority for approval.

4. **Specific Use**: The building must be used for temporary purposes such as entertainment, funeral, religious events, or construction-related activities.

5. **Insurance Requirement**: If the temporary building involves pilling or ground excavation, appropriate insurance coverage is mandatory.

6. **Compliance with Regulations**: The building must adhere to all relevant building regulations, local laws, and international standards.

7. **Demolition Requirement**: The temporary building must be demolished upon the expiration of the permitted period.

**Additional Considerations**:
- Further information may be needed regarding whether the building involves pilling or excavation to determine insurance requirements.
- The Relevant Authority may request detailed designs if deemed necessary, ensuring compliance with all applicable standards.
```

*Tokens: 1437, Time: 5.10s*

### Final Answer (Mistral)

```
The requirements for temporary buildings are:

1. Obtain Permission from the Relevant Authority, which may include specific conditions.
2. Validity Period: The permission is granted for a specified period, not exceeding one year.
3. Submission of Plans: A plan or sketch along with a written request must be submitted to the Relevant Authority for approval.
4. Specific Use: The building must be used for temporary purposes such as entertainment, funeral, religious events, or construction-related activities.
5. Insurance Requirement: If the temporary building involves pilling or ground excavation, appropriate insurance coverage is mandatory.
6. Compliance with Regulations: The building must adhere to all relevant building regulations, local laws, and international standards.
7. Demolition Requirement: The temporary building must be demolished upon the expiration of the permitted period.
```

*Tokens: 1764*

---

## Query: "What are the requirements for advertising boards and hoardings?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the requirements for advertising boards and hoardings based on the provided context. Let me go through each document step by step and see what information I can gather.

Starting with Document 1, it seems to list some fees associated with different types of advertising. There are digital and non-digital advertising boards, name boards, and gantries, each with their own rates per square meter. It also mentions land extent and some other activities, but I'm not sure how relevant that is to the question about requirements. Maybe the fees are part of the regulatory process, so I should note that there are costs involved, but I don't see any specific requirements here beyond the pricing.

Moving on to Document 2, this one talks about how billboards and hoardings should be erected and maintained. It says they shouldn't disturb vehicular movements or misguide road users. The colors, shapes, fonts, and symbols used shouldn't be similar to traffic signs. Also, they must be placed away from access road boundaries to avoid disrupting pedestrians and vehicles. So, this document is about placement and design restrictions to ensure safety and avoid confusion with traffic signs.

Document 3 discusses temporary advertisements for special events like festivals or commercial events. It mentions that temporary hoardings are allowed for a limited time and must be removed after the event, restoring the area to its original condition. If they aren't removed, there might be some repercussions, though the exact consequence isn't clear here. So, this is about temporary permits and the responsibility to remove structures after the event.

Document 4 seems to focus on the content and environmental impact of advertisements. It states that permissions won't be granted if the ads are detrimental to the area's environmental values or architectural character. Also, ads with obscene content are banned. This ties into the aesthetic and cultural considerations for allowing advertising in certain areas.

Document 5 has more specific rules. It says that no advertisement can cover more than 25% of a building's frontage. Also, they can't be supported by trees, utility poles, rocks, bridges, walls, vehicles, or traffic signs. Additionally, ads must be maintained in a pleasant and protective manner. There's also a mention of temporary permits for events, similar to Document 3.

Putting this all together, the requirements seem to cover several areas: placement, design, content, maintenance, and temporary permits. Placement-wise, they can't be too close to roads or on certain structures. Design-wise, they shouldn't mimic traffic signs. Content can't be obscene or harmful to the area's character. Maintenance is required to keep them looking good. Temporary ones need to be taken down after events.

I'm not sure if there are any other permits or approvals needed beyond what's mentioned. Also, the exact process for obtaining permission isn't detailed here. It would be helpful to know if there's a specific authority to approach or forms to fill out. Additionally, while Document 1 mentions fees, it's unclear if these are part of the permit process or separate charges.

Another point is the size restrictions. Document 5 mentions the 25% frontage rule, but I don't see any maximum size limits elsewhere. Maybe size is regulated through other means not covered here. Also, the environmental impact assessment seems important, but the documents don't specify how that's conducted or by whom.

In summary, the requirements are about where, how, and what can be advertised, ensuring safety, aesthetics, and environmental considerations. But some details like the permit process, size limits beyond frontage, and enforcement mechanisms aren't fully clear from the context provided.
</think>

The requirements for advertising boards and hoardings in Sri Lanka, based on the provided context, can be summarized as follows:

1. **Placement and Design Restrictions:**
   - **Proximity to Roads:** Advertisements must be erected away from access road boundaries to avoid disrupting pedestrian and vehicular movement.
   - **Design Considerations:** The color, shape, font, and symbols used must not resemble traffic signs to prevent confusion.

2. **Content Regulations:**
   - **Prohibited Content:** Advertisements containing obscene content or messages detrimental to the environmental or architectural character of the area are banned.

3. **Structural and Maintenance Requirements:**
   - **Support Structures:** Advertisements cannot be supported by trees, utility poles, rocks, bridges, walls, vehicles, or traffic signs.
   - **Maintenance:** Advertisements must be maintained in a pleasant and protective manner.

4. **Temporary Permits:**
   - **Duration and Removal:** Temporary advertisements for special events must be removed promptly after the event concludes, restoring the area to its original condition.

5. **Fees and Charges:**
   - **Costs:** There are specific fees for different types of advertising boards, indicating that cost is part of the regulatory process, though the exact permit process isn't detailed.

6. **Size and Coverage:**
   - **Frontage Limitation:** No advertisement can
```

*Tokens: 1769, Time: 4.53s*

### Final Answer (Mistral)

```
The requirements for advertising boards and hoardings in Sri Lanka, based on the provided context, can be summarized as follows:

1. **Placement and Design Restrictions:**
   - **Proximity to Roads:** Advertisements must be erected away from access road boundaries to avoid disrupting pedestrian and vehicular movement.
   - **Design Considerations:** The color, shape, font, and symbols used must not resemble traffic signs to prevent confusion.

2. **Content Regulations:**
   - **Prohibited Content:** Advertisements containing obscene content or messages detrimental to the environmental or architectural character of the area are banned.

3. **Structural and Maintenance Requirements:**
   - **Support Structures:** Advertisements cannot be supported by trees, utility poles, rocks, bridges, walls, vehicles, or traffic signs.
   - **Maintenance:** Advertisements must be maintained in a pleasant and protective manner.

4. **Temporary Permits:**
   - **Duration and Removal:** Temporary advertisements for special events must be removed promptly after the event concludes, restoring the area to its original condition.

5. **Fees and Charges:**
   - **Costs:** There are specific fees for different types of advertising boards, indicating that cost is part of the regulatory process, though the exact permit process isn't detailed.

6. **Size and Coverage:**
   - **Frontage Limitation:** No advertisement can cover more than 25% of a building's frontage. However, there are no maximum size limits elsewhere mentioned in the context provided.
```

*Tokens: 2354*

---

## Query: "What are the specifications for communication towers?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the specifications for communication towers based on the provided context. Let me go through each document step by step.

Starting with Document 1, it mentions that the specifications for communication towers are set out in Schedule 15. That's a key point. It also lists two main points: adherence to location-specific planning regulations and not disturbing the scenic beauty. So, the towers must comply with local regulations and environmental aesthetics.

Moving to Document 2, it talks about the application process for communication towers, including fees and recommendations from relevant institutions and the Telecommunication Regulatory Commission. While this is about the approval process, it doesn't directly give specs, but it's good to note the regulatory bodies involved.

Document 3 discusses indoor mobile infrastructure, which isn't directly about towers but shows that there are guidelines for communication infrastructure within buildings. Not super relevant here, but it shows a regulatory framework exists.

Document 4 seems to be about rainwater harvesting and groundwater infiltration. It mentions Schedule 15 again, which is about communication towers. However, the details here are more about storage and distribution, not directly the tower specs. Maybe it's a table of some sort, but the specifics are unclear.

Document 5 refers to Schedule 16, which deals with setbacks from electricity lines. It provides distances based on voltage levels. This is important for safety, ensuring towers are a safe distance from power lines to prevent hazards.

Putting it all together, the main source is Schedule 15 in Document 1, which outlines the specifications. It includes height limits (up to 100m for ground-based structures) and structural stability reports from engineers. Additionally, towers must follow local planning regulations and not harm the environment or scenic areas. Setbacks from power lines, as per Schedule 16, are also crucial for safety.

I think I've covered all the relevant parts. Now, I should structure this reasoning clearly, making sure to highlight where more info might be needed, like exact details from Schedules 15 and 16.
</think>

The specifications for communication towers are primarily outlined in Schedule 15 of the regulations, which includes height limits and structural requirements. Towers must adhere to local planning regulations and environmental considerations to preserve scenic beauty. Additionally, Schedule 16 provides safety guidelines for setbacks from electricity lines based on voltage levels. While the exact details of these schedules are not fully provided, they form the framework for compliance. Further information from the schedules would be needed for comprehensive specifications.
```

*Tokens: 1259, Time: 2.69s*

### Final Answer (Mistral)

```
The specifications for communication towers are primarily outlined in Schedule 15 of the regulations, which includes height limits and structural requirements. Towers must adhere to local planning regulations and environmental considerations to preserve scenic beauty. Additionally, Schedule 16 provides safety guidelines for setbacks from electricity lines based on voltage levels. While the exact details of these schedules are not fully provided, they form the framework for compliance. Further information from the schedules would be needed for comprehensive specifications.
```

*Tokens: 1546*

---

## Query: "What are the requirements for changing the use of a building?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the requirements for changing the use of a building based on the provided context. Let me go through each document step by step to understand what's needed.

Starting with Document 1, it mentions that if a building owner wants to use the building for a purpose different from what's specified in the CoC (Certificate of Conformity), they must notify the Authority and get approval. It also says that to change the use, an application using Form G is needed, along with copies of the Survey Plan, Building Plan, CoC, and payment. So, that's a clear requirement: notify the Authority, submit Form G with supporting documents.

Moving to Document 2, it talks about repairs, alterations, or additions needing a certificate from a Relevant Qualified Person. This person must confirm the building can handle any increased loads. Also, extensions must be clearly indicated, and parts to be removed marked. This seems relevant because changing use might involve structural changes, so this certificate and marking might be part of the process.

Document 3 states that no construction, alteration, or renovation can be done without a Development Permit. To get this permit, the developer must apply. So, changing the use might require a new Development Permit, especially if structural changes are involved.

Document 4 mentions that the Authority can revoke a CoC if there's a breach of permit terms, failure to comply with regulations, or misrepresentation. It also talks about submitting structural stability reports. This implies that when changing use, ensuring structural integrity is crucial, and non-compliance can lead to losing the CoC.

Document 5 says the Authority can revoke a Development Permit or CoC if violations are found. It also mentions that the CoC can be revoked for non-single unit residential buildings if they don't meet safety standards. So, changing use must maintain safety and structural standards.

Putting it all together, the requirements seem to include:

1. Notification to the Authority of the intended change.
2. Submission of Form G with necessary documents (Survey Plan, Building Plan, CoC, payment).
3. Obtaining approval from the Authority.
4. If structural changes are involved, getting a certificate from a Relevant Qualified Person.
5. Ensuring compliance with all regulations to avoid revocation of permits or certificates.

I'm not entirely sure if all these steps are mandatory for every type of use change or only when structural changes occur. Also, the context doesn't specify if there are different requirements for different types of buildings, like residential vs. commercial. More details on the building type and extent of changes would help clarify the exact requirements.
</think>

To change the use of a building, the following requirements must be met based on the provided context:

1. **Notification and Approval**: The building owner must notify the Relevant Authority of the intended change in use and obtain approval. This involves submitting an application using Form G.

2. **Submission of Documents**: The application must include copies of the approved Survey Plan, Building Plan, Certificate of Conformity (CoC), and payment of the required fees.

3. **Structural Assessment**: If the change of use involves structural alterations, a certificate from a Relevant Qualified Person is necessary to confirm the building's structural integrity and capacity to handle any increased loads.

4. **Compliance with Regulations**: The change must comply with all relevant building regulations and safety standards to avoid revocation of permits or certificates.

5. **Development Permit**: Depending on the nature of the change, a new Development Permit may be required, especially if construction, alteration, or renovation is involved.

**Considerations**:
- The context does not specify if requirements differ for building types (e.g., residential vs. commercial).
- The extent of structural changes may influence the necessity of certain steps.

In summary, changing the use of a building requires notification, submission of specific documents, possible structural assessment, and compliance with regulations to ensure safety and legal standards are met.
```

*Tokens: 1493, Time: 3.68s*

### Final Answer (Mistral)

```
To change the use of a building, the following requirements must be met based on the provided context:

1. **Notification and Approval**: The building owner must notify the Relevant Authority of the intended change in use and obtain approval. This involves submitting an application using Form G.

2. **Submission of Documents**: The application must include copies of the approved Survey Plan, Building Plan, Certificate of Conformity (CoC), and payment of the required fees.

3. **Structural Assessment**: If the change of use involves structural alterations, a certificate from a Relevant Qualified Person is necessary to confirm the building's structural integrity and capacity to handle any increased loads.

4. **Compliance with Regulations**: The change must comply with all relevant building regulations and safety standards to avoid revocation of permits or certificates.

5. **Development Permit**: Depending on the nature of the change, a new Development Permit may be required, especially if construction, alteration, or renovation is involved.

**Considerations**:
- The context does not specify if requirements differ for building types (e.g., residential vs. commercial).
- The extent of structural changes may influence the necessity of certain steps.
```

*Tokens: 1909*

---

## Query: "What provisions are made for qualified persons to certify building works?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out what provisions are made for qualified persons to certify building works based on the given context. Let me start by going through each document one by one to see what they say about certifications and who is responsible.

Looking at Document 1, it mentions that if applicable, the person needs to submit a certificate from a Relevant Qualified Person (RQP) as per Schedule 3. This RQP confirms that the development was done under their supervision, complies with the Approved Plan, and meets the Development Permit conditions. Additionally, another certificate from a Chartered Civil or Structural Engineer is required, though the exact details are cut off.

Moving to Document 2, it talks about submitting a certificate from an RQP who has examined the building and deems it capable of bearing increased or altered loads due to repairs, alterations, or additions. It also mentions marking extensions and removals in plans, but that's more about documentation than certification.

Document 3 seems to elaborate on the types of certificates needed. It mentions a certificate from a Chartered Civil or Structural Engineer regarding structural compliance. It also refers to another certificate from an RQP who is knowledgeable in various fields like mechanical, electrical, plumbing, etc. This suggests that different aspects of the building require different specialized certifications.

Document 4 has a regulation that non-private residential buildings not in Category B must be examined by an RQP every six months until completion. After each exam, a certificate is submitted. This indicates a regular inspection and certification process for larger projects.

Document 5 discusses insurance for works involving piling or excavation and allows early piling with prior approval. It doesn't directly talk about certification but implies that certain specialized works might require specific certifications, though it's not explicitly stated here.

Putting this all together, the provisions require multiple certifications from different qualified persons. There's a general RQP for overall compliance, a Chartered Engineer for structural aspects, and specialized RQPs for systems like electrical and plumbing. The process involves initial certification, ongoing inspections every six months for certain projects, and submission of these certificates at various stages.

I'm noticing that while the context provides a framework, some details are missing, like the exact qualifications of RQPs or the consequences of non-compliance. Also, how these different certificates interact or if there's an overarching authority coordinating them isn't clear. More information on Schedule 3 would help understand the specifics of who qualifies as an RQP and their roles.
</think>

The provisions for qualified persons certifying building works are structured as follows:

1. **General Certification by Relevant Qualified Persons (RQP):**
   - An RQP, as defined in Schedule 3, must certify that the development complies with the Approved Plan and Development Permit conditions. This includes supervision of the development.

2. **Structural Certification:**
   - A Chartered Civil or Structural Engineer must provide a certificate confirming structural compliance with the Approved Plans and Development Permit.

3. **Specialized Systems Certification:**
   - Additional certifications are required from RQPs specializing in areas such as mechanical, electrical, plumbing, and others, ensuring each system meets necessary standards.

4. **Regular Inspections:**
   - For projects beyond single-unit private residences not in Category B, RQPs must conduct six-month inspections until completion, submitting certificates after each inspection.

5. **Documentation and Submission:**
   - All certificates must be submitted as part of the approval process, with ongoing submissions required for larger projects.

**Ambiguities and Areas Needing Clarification:**
- The exact qualifications and roles of RQPs as per Schedule 3 are not detailed.
- The coordination between different certifying authorities and potential consequences of non-compliance are not specified.

This framework ensures comprehensive certification across various aspects of building works, though additional details would enhance clarity and compliance understanding.
```

*Tokens: 1450, Time: 3.62s*

### Final Answer (Mistral)

```
The provisions for qualified persons to certify building works include:

1. General certification by Relevant Qualified Persons (RQP) who confirm that the development complies with the Approved Plan and Development Permit conditions.
2. Structural certification from a Chartered Civil or Structural Engineer ensuring structural compliance with the Approved Plans and Development Permit.
3. Specialized systems certifications from RQPs specializing in areas such as mechanical, electrical, plumbing, etc., to ensure each system meets necessary standards.
4. Regular inspections by RQPs every six months for non-private residential buildings not in Category B until completion.
5. Submission of all certificates as part of the approval process and ongoing submissions required for larger projects.
```

*Tokens: 1804*

---

## Query: "What happens if the services of a Qualified Person are terminated during construction?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out what happens if the services of a Qualified Person are terminated during construction in Sri Lanka. I'm not very familiar with the specific regulations, but I'll try to piece it together using the provided context.

First, I'll look through the documents to find any relevant information. Document 1 mentions Regulation 111, which talks about the termination of a Qualified Person's services. It says the Developer must notify the Relevant Authority within 14 days. That seems important because it's a legal requirement, so missing this deadline could have consequences.

Document 2 continues from Document 1. It states that after termination, no building work can commence, continue, or resume unless another Qualified Person is appointed and notified to the Authority. So, the construction can't just proceed without a Qualified Person; they need to get a replacement. Also, the terminated person has to provide a summary report and a certification of the work done so far. That makes sense because the Authority needs to know the status of the project when the Qualified Person leaves.

Looking at the other documents, like Document 3, it talks about certificates for repairs and alterations, but that might not be directly relevant here. Document 4 discusses mechanical systems and discontinuing services, which also doesn't seem related. Document 5 is about Development Permits and their validity, which is a different aspect but might tie in if the termination affects the project timeline beyond the permit's validity.

So, putting it together: If a Qualified Person is terminated, the Developer has two weeks to inform the Authority. Work must stop until a new Qualified Person is appointed and the Authority is notified. The former Qualified Person must provide a report and certification. If the Developer doesn't appoint someone new, the project can't continue, which could lead to delays or legal issues.

I'm not sure about the consequences of not notifying the Authority within 14 days. Maybe there are penalties or the permit could be revoked. Also, if the project goes beyond the Development Permit's three-year validity, the Developer might need to apply for a new permit, as mentioned in Document 5.

I think I've covered the main points, but I'm not entirely sure if there are more steps or if other regulations come into play. Maybe the new Qualified Person needs to review the previous work, but that's not specified here. Also, the process for appointing a new person isn't detailed, so I assume it's similar to the initial appointment process.
</think>

If the services of a Qualified Person are terminated during construction in Sri Lanka, the following steps and consequences ensue based on the provided context:

1. **Notification Requirement**: The Developer must notify the Relevant Authority in writing within 14 days of the termination. This is a legal requirement, and failure to comply may result in unspecified penalties or complications.

2. **Work Stoppage**: Construction must cease immediately upon termination. No work can commence, continue, or resume until a new Qualified Person is appointed.

3. **Appointment of a New Qualified Person**: The Developer must appoint another Qualified Person and notify the Relevant Authority in writing. This ensures that the project continues under the required professional supervision.

4. **Reporting and Certification**: The terminated Qualified Person must provide a summary report and certification of the work completed up to the point of termination. This documentation is crucial for the Authority to assess the project's status.

5. **Potential Consequences of Non-Compliance**: If the Developer fails to appoint a new Qualified Person, the project may face delays or legal issues, potentially leading to the revocation of the Development Permit.

6. **Development Permit Considerations**: If the project timeline exceeds the three-year validity of the Development Permit, the Developer must apply for a new permit, as extensions beyond this period are not automatically granted.

In summary, termination of a Qualified Person's services necessitates prompt notification, work stoppage, and the appointment of a replacement to avoid legal repercussions and ensure compliance with regulations.
```

*Tokens: 1495, Time: 3.63s*

### Final Answer (Mistral)

```
If the services of a Qualified Person are terminated during construction in Sri Lanka, the following steps and consequences ensue based on the provided context:

1. Notification Requirement: The Developer must notify the Relevant Authority in writing within 14 days of the termination. This is a legal requirement, and failure to comply may result in unspecified penalties or complications.
2. Work Stoppage: Construction must cease immediately upon termination. No work can commence, continue, or resume until a new Qualified Person is appointed.
3. Appointment of a New Qualified Person: The Developer must appoint another Qualified Person and notify the Relevant Authority in writing. This ensures that the project continues under the required professional supervision.
4. Reporting and Certification: The terminated Qualified Person must provide a summary report and certification of the work completed up to the point of termination. This documentation is crucial for the Authority to assess the project's status.
5. Potential Consequences of Non-Compliance: If the Developer fails to appoint a new Qualified Person, the project may face delays or legal issues, potentially leading to the revocation of the Development Permit.
6. Development Permit Considerations: If the project timeline exceeds the three-year validity of the Development Permit, the Developer must apply for a new permit, as extensions beyond this period are not automatically granted.
```

*Tokens: 1989*

---

## Query: "What are the requirements for developments near expressways?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the requirements for developments near expressways in Sri Lanka based on the provided context. Let me go through each document one by one and see what relevant information I can find.

Starting with Document 1, it talks about the minimum turning radius and margin lanes. If the turning radius can't be provided, a margin lane of at least 3.0m must be provided within the site. This lane needs to be designed to minimize traffic impact and must be approved by the Traffic Planning Committee. So, this seems relevant because developments near expressways would need to consider access and traffic flow.

Moving to Document 2, it mentions widening access roads according to Street Lines or equally from the center line. The land for widening must be gifted to the Relevant Authority without payment, using a deed of gift as per Annexure 6 of Form B. This suggests that if a development near an expressway requires road widening, the developer must donate the necessary land. That's an important requirement.

Document 3 discusses the width of access roads based on the number of lots served. It also mentions that when a main access road is 9.0m or more, lots beyond a certain point might have specific requirements. This could relate to how developments near expressways need to have adequate access roads, possibly wider than usual, to handle the traffic generated.

Document 4 refers to a study area for traffic impact assessments. It says the study area is a 1 km radius around the project site, considering surrounding developments and traffic generation. It also mentions that traffic surveys should be done within 6 months of the TIA report. This is crucial because any development near an expressway would likely require a thorough traffic impact study to assess how it affects the surrounding area and the expressway itself.

Document 5 talks about the Relevant Authority issuing Development Permits with guidelines and conditions. It also mentions a Certificate of Conformity (CoC) being granted if the Development Permit's requirements are met. This indicates that any development near an expressway must obtain the necessary permits and certifications, adhering to specific guidelines.

Putting this all together, the requirements would involve traffic impact assessments, providing margin lanes if turning radius isn't possible, ensuring access roads are adequately wide, gifting land for road widening if needed, and obtaining the proper permits and certificates from the Relevant Authority. I should also note that some areas might need more specific information, like exact distance thresholds for expressway proximity or detailed design standards for access roads and margin lanes.

I think I've covered all the documents, but I'm not entirely sure if there are more specific regulations about distance from the expressway or noise and environmental considerations. Those might be important too, but they aren't mentioned in the provided context. Also, the exact process for getting approvals from the Traffic Planning Committee isn't detailed here, so that's another area where more info would be helpful.
</think>

To determine the requirements for developments near expressways in Sri Lanka, we must consider several key legal principles and regulations derived from the provided context. Below is a structured analysis of the applicable legal framework and how it applies to developments near expressways.

---

### 1. **Key Legal Principles and Regulations**
The following principles and regulations are relevant to developments near expressways:

1. **Traffic Impact Assessments (TIA) and Study Area Requirements (Document 4):**
   - Developments near expressways must consider their impact on traffic flow and surrounding infrastructure.
   - A study area of 1 km radius from the project boundary must be analyzed, including surrounding developments and traffic generation.
   - Traffic surveys must be conducted within 6 months of the TIA report to ensure data accuracy.

2. **Access Road Requirements (Documents 2 and 3):**
   - The width of access roads must be determined based on the number of lots served, as specified in Form B (Schedule 4).
   - For large subdivisions requiring main access roads of 9.0m or more, specific lot arrangements beyond the main road may apply.
   - Road widening, if required, must be done according to Street Lines or equally from the center line, with land gifted to the Relevant Authority without monetary consideration.

3. **Margin Lanes for Turning Radius (Document 1):**
   - If the minimum turning radius cannot be provided, a margin lane of at least 3.0m must be provided within the site.
   - Such lanes must be designed to minimize traffic impact and require approval from the Traffic Planning Committee.

4. **Development Permits and Certificates of Conformity (Document 5):**
   - Developments near expressways must obtain a Development Permit from the Relevant Authority, which includes guidelines and conditions for subdivision, amalgamation, or other developments.
   - A Certificate of Conformity (CoC) must be granted by the Relevant Authority after ensuring compliance with the Development Permit requirements.

---

### 2. **Application to Developments Near Expressways**
The above principles apply to developments near expressways in the following
```

*Tokens: 1728, Time: 4.46s*

### Final Answer (Mistral)

```
The requirements for developments near expressways in Sri Lanka include conducting a Traffic Impact Assessment (TIA) with a study area of 1 km radius from the project boundary, considering surrounding developments and traffic generation. Access roads must be designed based on the number of lots served, with specific lot arrangements for large subdivisions requiring main access roads of 9.0m or more. If the minimum turning radius cannot be provided, a margin lane of at least 3.0m must be provided within the site, subject to approval from the Traffic Planning Committee. Developments near expressways must obtain a Development Permit with guidelines and conditions for subdivision, amalgamation, or other developments, along with a Certificate of Conformity (CoC) after ensuring compliance with the Development Permit requirements. Road widening, if necessary, must be done according to Street Lines or equally from the center line, with land gifted to the Relevant Authority without monetary consideration.
```

*Tokens: 2089*

---

## Query: "What is the requirement for display notices during construction?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the requirements for displaying notices during construction based on the provided context. Let me go through each document step by step to see what information I can gather.

Starting with Document 1, I see that it talks about advertisements, billboards, and hoardings. Specifically, point (14) mentions that display of advertisements or notices may be permitted for a limited period for special events, cultural or religious festivals, etc. But I'm not sure if this applies to construction notices or just temporary ads for events.

Moving on to Document 2, this seems more relevant. Point (3) states that for developments over 900 m² or buildings of G+4 or more stories, a notice board of specific dimensions (2.5m x 1.5m) must be displayed. It should include details like owner, contractor, architect, approved plan number, and date. This sounds like a clear requirement for construction sites. Point (4) mentions temporary buildings but doesn't directly talk about notices, so maybe not as relevant.

Document 3 again refers to temporary hoardings for special events and the need to remove them after the event. This seems similar to Document 1, so probably not about construction notices but more about event-related signage.

Document 4 and Document 5 discuss the placement and restrictions on advertisements to avoid obstructing roads and pedestrian movement. They mention not covering more than 25% of a building's frontage and not obstructing doors or windows. While important for general signage, it doesn't specifically address construction notices.

Putting it all together, the key point is in Document 2, point (3). It clearly outlines the requirement for a notice board in larger construction projects. However, I'm not sure if smaller projects have any requirements. The context doesn't specify, so maybe the rules only apply to bigger developments. Also, the content of the notice is detailed, which is helpful, but there's no mention of how long the notice must be displayed or any maintenance requirements beyond what's in other documents about general signage upkeep.

I think the main requirement is the notice board for larger projects as per Document 2. For smaller projects, the context doesn't provide info, so maybe no specific notice is needed, or it's covered under general advertising rules. But since the question is about construction, the answer likely hinges on Document 2's provisions.
</think>

The requirement for displaying notices during construction is primarily outlined in Document 2. For developments exceeding 900 square meters or buildings of G+4 floors or more, a notice board of 2.5m x 1.5m must be erected. This board should include the owner's, contractor's, and architect's details, along with the approved plan number and date. This requirement ensures transparency and compliance for larger projects. Smaller projects may not have specific notice requirements under the provided context, suggesting that such notices might not be mandatory or may fall under general signage regulations. Thus, the main requirement applies to larger developments as specified.
```

*Tokens: 1326, Time: 3.01s*

### Final Answer (Mistral)

```
The requirement for displaying notices during construction is primarily outlined in Document 2. For developments exceeding 900 square meters or buildings of G+4 floors or more, a notice board of 2.5m x 1.5m must be erected. This board should include the owner's, contractor's, and architect's details, along with the approved plan number and date.
```

*Tokens: 1517*

---

## Query: "How long is a Development Plan valid?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out how long a Development Plan is valid under Sri Lankan building regulations. Let me start by going through the provided documents to find relevant information.

Looking at Document 2, it mentions that the validity period of a Development Permit expires after three years. That seems straightforward. Then, in Document 3, it says that if the developer hasn't completed the project due to unforeseeable circumstances, they can apply for a renewal for up to two more years. So, the initial permit is three years, with a possible extension of two years.

Wait, but the question is about a Development Plan, not a Development Permit. I should check if these terms are used interchangeably or if they refer to different things. In Document 1, a Development Guide Plan is defined as a layout with specific guidelines, while a Development Permit is a permit with terms and conditions. So, they might be related but not the same.

If the question is about the Development Plan's validity, the context provided doesn't explicitly state its duration. It only talks about the Development Permit. So, maybe the answer depends on whether the Development Plan's validity is tied to the Permit. If the Permit is for three years, perhaps the Plan is valid as long as the Permit is active, including any extensions.

But without explicit information linking the Plan's validity to the Permit, I can't be certain. The context might be missing some details. I should note that the provided documents don't specify the Development Plan's validity period, only the Permit's. Therefore, more information would be needed to accurately answer the question about the Development Plan.
</think>

The validity of a Development Plan under Sri Lankan building regulations isn't explicitly stated in the provided context. The documents discuss the Development Permit, which is valid for three years with a possible two-year extension under certain conditions. However, since the question pertains to the Development Plan, which is defined differently as a layout with guidelines, its validity isn't clear from the given information. More details are needed to determine the Development Plan's validity period. 

**Answer:**
The context provided does not specify the validity period of a Development Plan. It details the validity of a Development Permit, which is three years with a possible two-year extension. To determine the Development Plan's validity, additional information linking it to the Permit or other regulations is required.
```

*Tokens: 1165, Time: 2.99s*

### Final Answer (Mistral)

```
The context provided does not specify the validity period of a Development Plan. It details the validity of a Development Permit, which is three years with a possible two-year extension. To determine the Development Plan's validity, additional information linking it to the Permit or other regulations is required.
```

*Tokens: 1325*

---

## Query: "I need to erect a temporary shed for a religious festival that will last for 14 days. What specific approval process applies, and what documentation must I submit?"

### Legal Reasoning (DeepSeek-R1)

```
<think>
Okay, so I need to figure out the approval process and documentation required for erecting a temporary shed for a religious festival that lasts 14 days in Sri Lanka. Let me go through the provided documents step by step to understand what's needed.

First, looking at Document 1, it talks about Temporary Buildings. It defines a Temporary Building as a shed for religious purposes, among other things. It also mentions that the Relevant Authority can grant permission for up to one year, but in this case, the shed is only needed for 14 days, which is well within the one-year limit. So, I know I need to apply for a Temporary Building permission.

Next, Document 2, section 109, says that for any new development related to religious purposes, consent must be obtained from the public living within a 500-meter radius through the Grama Niladhari. That means I need to get the community's consent, probably through some kind of public notice or meeting. I should check if there's a specific process for this, like notifying the Grama Niladhari and getting their approval.

Document 3 discusses temporary hoardings and notices for special events. It says these can be permitted for a limited time and must be removed after the event. Although this is about hoardings, it might imply that temporary structures like my shed should also be removed after the festival. So, I need to ensure that the shed is taken down after 14 days and the area is restored.

Document 4 mentions that the Relevant Authority may permit a Temporary Building for a specific period, as per Regulation 106. It also talks about insurance for buildings involving piling or excavation. Since my shed is temporary and probably doesn't require deep foundations, I might not need insurance, but I should confirm if any structural aspects require it.

Document 5 refers to solid waste management. It says that after construction, the developer must submit a solid waste proposal. Since my shed is temporary, I might need to show how I'll manage waste during and after the festival. Maybe a simple plan outlining disposal methods would suffice.

Putting this all together, I think the steps are: apply for Temporary Building permission from the Relevant Authority, obtain public consent via the Grama Niladhari, ensure the shed is removed after the festival, and possibly submit a waste management plan. I should also check if any other documents like site plans or structural certificates are needed, even if it's temporary.

I'm a bit unsure about the exact process for public consent. Do I just inform the Grama Niladhari, or do I need to hold a meeting? Also, are there specific forms for the Temporary Building application, and what details should be included in the waste management plan? I might need to contact the local authority to clarify these points.
</think>

To erect a temporary shed for a 14-day religious festival in Sri Lanka, follow these steps:

1. **Temporary Building Permission:**
   - Apply to the Relevant Authority for a Temporary Building permit as defined in Document 1. Ensure the application specifies the shed's purpose, duration, and compliance with regulations.

2. **Public Consent:**
   - Obtain consent from residents within a 500-meter radius through the Grama Niladhari, as per Document 2. This may involve notifying the community and possibly attending a meeting.

3. **Removal and Restoration:**
   - Commit to dismantling the shed after the festival and restoring the site, adhering to the guidelines in Document 3.

4. **Structural and Safety Considerations:**
   - Check if insurance is needed, though likely unnecessary for a simple shed. Ensure the structure meets safety standards.

5. **Waste Management Plan:**
   - Submit a plan outlining waste disposal during and after the festival, as per Document 5.

6. **Documentation:**
   - Prepare a site plan, structural details if required, and any other forms specified by the Relevant Authority.

7. **Clarifications:**
   - Contact local authorities to confirm specific requirements, such as the process for public consent and necessary documentation details.

By following these steps, you can ensure compliance with Sri Lankan regulations for your temporary shed.
```

*Tokens: 1555, Time: 3.59s*

### Final Answer (Mistral)

```
To erect a temporary shed for a 14-day religious festival in Sri Lanka, you will need to follow the approval process and submit the required documentation as outlined below:

1. **Temporary Building Permission:**
You must apply to the Relevant Authority for a Temporary Building permit. The application should specify the purpose of the shed (religious), the duration of use (14 days), and compliance with regulations.
2. **Public Consent:**
Obtain consent from residents within a 500-meter radius through the Grama Niladhari, as per Document 2. This may involve notifying the community and possibly attending a meeting.
3. **Removal and Restoration:**
Commit to dismantling the shed after the festival and restoring the site, adhering to the guidelines in Document 3.
4. **Structural and Safety Considerations:**
Check if insurance is needed, though likely unnecessary for a simple shed. Ensure the structure meets safety standards.
5. **Waste Management Plan:**
Submit a plan outlining waste disposal during and after the festival, as per Document 5.
6. **Documentation:**
Prepare a site plan, structural details if required, and any other forms specified by the Relevant Authority.
7. **Clarifications:**
Contact local authorities to confirm specific requirements, such as the process for public consent and necessary documentation details.

By following these steps, you can ensure compliance with Sri Lankan regulations for your temporary shed.
```

*Tokens: 2028*

---

