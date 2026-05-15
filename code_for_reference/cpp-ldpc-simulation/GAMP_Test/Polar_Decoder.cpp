#include "Polar_Encoder.h"
#include "SSCL.h"
#include "SCAN.h"
#include "Polar_Function.h"
#include <xmmintrin.h>
#include <immintrin.h>
#include <vector>
#include <fstream>
#include <ctime>
#include <functional>
#include <algorithm>
#include <queue>
#include <time.h>

vector<float> abs_LLR_it(32768);
vector<int> index_LLR_it(32768);

#define hard(n) (n>=0?0:1)
#define alph 0.5
#define mthres(n) (abs(n)+log(1 + exp(-alph*abs(n))) / alph)
#define hthres(n) (log(1 + exp(-alph*abs(n))) / alph)

int R1F[1024] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

extern int Count;
extern int Count_info;
extern int l;

int u_last1[32768];
int u_last2[32768];

int sum_temp[1024];

float W10[256] = { -0.00042297831, 0.057407681, -0.028156461, 0.15360056, 0.055386160, 0.060348824, -0.0051058838, 0.26368442, -0.048220970, 0.082784072, -0.096483685, 0.13267921, 0.13501205, 0.14699258, 0.052232396, 0.13790198, 0.014552412, 0.10605203, -0.011732006, 0.17333393, 0.15199250, 0.17321838, 0.18334879, 0.26403230, -0.0061619971, 0.20157517, 0.25300801, 0.21202663, 0.083467856, 0.18735115, 0.065746427, 0.18318909, 0.030944195, 0.078793257, -0.016632264, 0.21731581, 0.084798619, 0.10639451, 0.065136515, 0.18960637, 0.084134266, 0.16028892, 0.14161173, 0.19104066, 0.21940306, 0.22247341, 0.052944545, 0.17007302, 0.071825683, 0.11259546, 0.16952822, 0.29193208, 0.15066044, 0.28443658, 0.16006035, 0.11817912, 0.11521586, 0.33517915, 0.18094093, 0.17136498, 0.19423540, 0.080359370, 0.12506741, 0.22018120, 0.044295974, 0.13430959, 0.12112667, 0.16222696, 0.046232380, 0.14938547, -0.022818016, 0.26235938, 0.073781669, 0.14313991, 0.19302504, 0.19045700, 0.13529092, 0.13864771, 0.091921568, 0.13741341, 0.079197787, 0.14672449, 0.094836220, 0.19663578, 0.13906243, 0.13243099, 0.094418801, 0.18719973, 0.15936016, 0.16856872, 0.17850679, 0.12060603, 0.22554056, 0.12250718, 0.22280568, 0.12437319, 0.056223363, 0.15612143, 0.10909915, 0.20485875, 0.089062877, 0.19098827, 0.11942779, 0.23088191, 0.0087745646, 0.16292121, -0.0038293635, 0.20287830, 0.20430720, 0.18564801, 0.14460449, 0.14861098, 0.029513486, 0.10871424, 0.0038912129, 0.23549135, 0.24551405, 0.18070365, 0.069516003, 0.11535344, 0.12202431, 0.065469541, 0.12703513, 0.10248551, 0.049391456, 0.13999273, 0.049753878, 0.074933611, 0.076988004, 0.11007751, 0.070350118, 0.14841978, 0.090517633, 0.10365136, 0.054796651, 0.11175932, 0.15870078, 0.22406043, 0.062968671, 0.22433275, 0.18496045, 0.21486671, 0.080211937, 0.14343129, 0.099648051, 0.17571296, 0.066276513, 0.16606857, 0.17697738, 0.16379508, 0.047531419, 0.14304970, 0.18172529, 0.12491643, 0.070346057, 0.19195195, 0.11023440, 0.089031160, 0.053985856, 0.27477089, 0.16083612, 0.13941805, 0.11413468, 0.11628301, 0.087050319, 0.071132615, 0.092189766, 0.21928796, 0.16752714, 0.12868738, 0.10292279, 0.15984307, 0.18708403, 0.16463137, 0.13780002, 0.12214648, -0.010280411, 0.065320484, 0.013877292, 0.18106882, 0.085369878, 0.20867589, 0.13330284, 0.16836843, 0.13484947, 0.20452796, 0.10270520, 0.19847991, 0.056922391, 0.16018476, 0.14330314, 0.10804275, 0.14894257, 0.11774050, 0.19167832, 0.13230532, 0.052742708, 0.15085869, 0.11566019, 0.14150895, 0.15365548, 0.073978275, 0.15523499, 0.066698000, 0.051367819, 0.14093198, 0.090099432, 0.16635497, -0.076027453, 0.25094551, 0.12728693, 0.13343321, 0.10306837, 0.15799305, 0.057762455, 0.13524422, 0.10479423, 0.18310305, 0.10446735, 0.18845108, -0.0078764977, 0.15854609, 0.24039342, 0.13025913, 0.15398626, 0.11369197, 0.057416342, 0.18536171, 0.16711032, 0.13391748, -0.010437805, 0.11678432, 0.021476625, 0.15284520, 0.10691804, 0.13965717, 0.019003548, 0.098758385, 0.085799441, 0.15550265, 0.066800460, 0.063730732, 0.096914053, 0.11097316, 0.029404437, 0.17329402, 0.097339869, 0.23423591, 0.085799634, 0.10497133, 0.092888370, 0.093784958, 0.22066039, 0.18161696, 0.13049011, 0.18936908 };
float W11[256] = { 0.098985262, 0.19524059, 0.15835243, 0.18166980, 0.20334238, 0.045262408, 0.15329951, 0.058825705, 0.17433979, 0.14899004, 0.070394114, -0.10172635, 0.10130119, 0.028813623, 0.0017026006, -0.047808878, 0.24568093, 0.17875329, 0.19566344, 0.023338187, 0.19072199, -0.021594800, 0.033982076, 0.018635366, 0.13638543, 0.10702421, 0.025348812, -0.066357598, 0.056526814, -0.13355520, -0.11204620, -0.080353834, 0.28260019, 0.072644100, 0.14589061, 0.022480398, 0.24600329, -0.052699748, 0.0061721215, -0.020909309, 0.22100762, -0.10509210, 0.10430966, -0.019434934, 0.067042992, 0.096695520, -0.11255264, -0.049697619, 0.21265431, 0.048045993, 0.075464338, 0.079547025, 0.093402490, 0.025548650, -0.19206689, -0.028150922, 0.071690187, -0.049333520, 0.057837818, 0.0054421602, 0.010651440, -0.044243004, -0.075725779, -0.13054039, 0.24394631, 0.093052298, 0.14274491, 0.14895840, 0.23848036, 0.043491330, -0.020246241, -0.056418829, 0.27877218, 0.019468559, 0.011260434, -0.012792160, 0.048867337, 0.013578432, 0.023165023, -0.051700499, 0.094243027, -0.035641558, 0.0093270009, -0.074863620, 0.0015091089, -0.080218509, -0.13261618, -0.17514987, 0.052978434, -0.0087126577, 0.080899894, -0.015460613, -0.064601921, 0.070507310, -0.053981606, -0.039057486, 0.14614058, 0.053862520, 0.047978729, -0.12802444, 0.13074847, 0.019868562, -0.037397079, -0.052263461, 0.064823724, 0.014658197, -0.11509239, -0.041726045, -0.0023736444, -0.056240249, 0.042492587, 0.00089978741, 0.039173543, -0.051104937, -0.0070302146, -0.076898888, 0.030468598, -0.088203833, -0.067104347, -0.020189226, 0.068006061, 0.065260306, 0.077670559, 0.055378150, 0.084854640, -0.046188172, 0.0024158689, 0.030416109, 0.17373180, 0.21749108, 0.089736238, 0.10943554, 0.20784971, 0.092409156, 0.091228426, -0.011736444, 0.18931861, 0.15476444, 0.017797764, 0.058525112, 0.017529001, 0.0020101322, -0.017570335, -0.095803842, 0.21514598, 0.066544838, 0.041623108, -0.016615881, 0.071743838, 0.043457966, -0.034125615, -0.0081218239, 0.10057630, 0.082086541, -0.055643912, -0.043105558, -0.072201267, -0.092509761, -0.074532323, 0.015455874, 0.072898351, 0.036879383, 0.073131330, 0.066562533, 0.12010263, 0.11503010, 0.080881201, 0.0031220831, 0.0015001313, 0.011771576, 0.010809463, 0.035439771, -0.087111287, -0.0023315719, -0.040657278, 0.15618379, 0.10588278, -0.11099581, -0.0039189481, 0.059060339, -0.11177916, -0.077441111, 0.019347206, 0.016260462, -0.0079015289, 0.089342825, -0.0098372838, 0.015303078, 0.010357518, -0.024049649, -0.028217129, -0.039988194, 0.11863333, 0.097224087, 0.088433497, 0.12771149, 0.033442657, -0.024350340, -0.042786002, 0.048215508, 0.045443136, 0.060365736, -0.010933376, 0.058057379, 0.059052456, 0.060923200, -0.0096185822, -0.049528003, 0.13735083, 0.047276177, 0.064549834, -0.025596932, 0.011017174, 0.033303563, 0.031478152, -0.026402621, 0.035048079, 0.036365267, -0.053925581, -0.0017401561, -0.052071001, 0.038455542, 0.011701216, 0.032073304, 0.12042059, 0.13918151, -0.019404819, 0.023034496, 0.12275274, 0.012597458, -0.10386716, -0.093168624, 0.012662943, 0.058065865, 0.077711277, 0.044829264, -0.051756255, 0.051802725, -0.017203426, 0.076010101, 0.074959040, 0.061505653, 0.058830913, 0.097702406, -0.029818591, 0.088606946, 0.014227658, 0.066046141, 0.16941993, 0.056312233, 0.044544116, 0.097640306, 0.17156138, -0.012790351, 0.10411718, 0.12305800 };
float W20[256] = { -0.46094015, 0.43633559, 0.022296075, 0.25631642, -0.018027358, 0.19720726, 0.057470251, 0.29679355, -0.37860468, 0.40261117, 0.076738670, 0.23541228, -0.0045512524, 0.35817832, 0.021034814, 0.082600988, -0.15376347, 0.24382685, 0.0069143083, 0.42378885, -0.13377011, 0.21589592, 0.20151091, 0.25737649, -0.095261648, 0.30979088, 0.0037363016, 0.42698896, 0.099637061, 0.36561009, 0.28877440, 0.29150182, -0.33111820, 0.44688696, -0.048101101, 0.19754727, -0.079659760, 0.067960232, 0.10801049, 0.36509436, -0.16001377, 0.29516000, 0.27675185, 0.30505925, 0.12946127, 0.48682177, 0.13769348, 0.83176196, -0.16164851, 0.27941334, -0.16261104, 0.27135921, -0.054682016, 0.27049544, 0.045592703, 0.60647070, -0.022311836, 0.20561817, -0.027135666, 0.49367788, 0.045436386, 0.43732280, 0.27178779, -0.063464642, -0.15659569, 0.21324554, -0.017228909, 0.34533492, 0.049284309, 0.31045160, 0.20098156, 0.28658617, -0.10567973, 0.28203493, 0.16830952, 0.25299793, 0.040473066, 0.36019593, 0.27403685, 0.27428377, -0.43289641, 0.33395451, 0.063137144, 0.41285050, -0.059790339, 0.34644169, 0.16243134, 0.46819457, -0.16500768, 0.29458413, -0.030732343, 0.69169539, 0.084734038, 0.67963201, -0.0027146093, 1.5020412, -0.24736468, 0.38736656, 0.28031629, 0.22122149, 0.033857808, 0.34718046, 0.10471033, 0.48561203, -0.031995453, 0.26579887, 0.047270060, 0.25940242, 0.072805136, 0.21325043, 0.055155717, 1.0254667, 0.059419252, 0.20034334, 0.11770073, 0.24028075, 0.19358698, 0.40231320, -0.029935751, 1.1144266, -0.042423341, 0.23133388, 0.053159289, 0.71413803, 0.13774607, 0.48589689, 0.41059747, 1.3823838, -0.29642272, 0.37729248, 0.10248443, 0.26368278, -0.049656890, 0.38513362, 0.21225637, 0.51812160, -0.25120482, 0.38146812, 0.090317219, 0.47070459, 0.034096252, 0.45469230, 0.32072768, 0.32117817, -0.30019778, 0.39217141, 0.089572765, 0.48831657, 0.033976641, 0.30087712, 0.29329267, 0.23880963, 0.062119145, 0.37397963, 0.15833141, 0.45111907, -0.0078640897, 0.32996494, 0.29893214, 0.54734182, -0.12205207, 0.55560136, -0.074297853, 0.30077282, 0.14459528, 0.35767430, 0.38997039, 0.31666556, -0.013474558, 0.42147827, 0.20676662, 0.23957373, 0.16191958, 0.58645815, 0.40696186, 0.86176884, -0.088817053, 0.19899385, -0.053819180, 0.40697804, 0.073771589, 0.42639428, 0.11543900, 0.75730419, 0.10585290, 0.41635475, 0.027400706, 0.66674829, -0.052973259, 0.44489783, -0.18416621, 1.6968349, 0.065487139, 0.45850888, 0.032583121, 0.50613797, 0.28126201, 0.51343107, 0.39728373, 0.43663642, -0.19387786, 0.50812751, -0.013331473, 0.52760583, -0.019827653, 0.51447433, 0.27779225, 0.37374693, -0.063735835, 0.30640569, 0.16108960, 0.57551575, 0.11396831, 0.42225060, 0.37151954, 0.33364624, -0.039812881, 0.25592792, 0.16328400, 0.58290660, 0.17992781, 0.64304167, -0.029518921, 1.3658801, 0.056326289, 0.44265467, 0.072782889, 0.33243474, 0.090638012, 0.47599763, 0.38775465, 0.35415563, 0.010042867, 0.42752618, 0.18178156, 0.52351743, 0.10895256, 0.45754698, -0.072498061, 0.95603675, 0.19893272, 0.35422969, 0.13746022, 0.72738075, 0.16418391, 0.32888731, 0.010066294, 0.71574652, 0.084842086, 0.34070107, 0.19822846, 0.65211928, 0.12195977, 0.40540144, -0.11541974, 1.8074192 };
float W21[256] = { 0.60488403, 0.53642207, 0.26107466, 0.68862152, 0.36960429, 0.51716536, 0.42891157, 0.40657640, 0.073780917, 0.60119331, 0.14887646, 0.42332539, 0.17823312, 0.43155026, 0.25863418, 0.13666210, 0.25978711, 0.51055187, 0.14452834, 0.46861315, 0.19699614, 0.40814909, 0.28060463, 0.49444735, 0.31447893, 0.41685310, 0.19455081, 0.73380339, 0.11620095, 0.67348880, 0.26733324, 0.12198652, 0.26382184, 0.50727260, 0.42934939, 0.47257751, 0.19843926, 0.39650881, 0.29067644, 0.41362226, 0.14841887, 0.54205287, 0.22606644, 0.34175795, 0.083373584, 0.68412423, 0.50149673, 0.83556211, 0.31046414, 0.49269927, 0.087682292, 0.60876447, 0.11353434, 0.41359252, 0.28153211, 0.81890994, 0.10285025, 0.33350566, 0.16654414, 0.70130557, 0.012133623, 0.69474381, 0.16795513, 0.016274713, 0.23553702, 0.47588557, 0.10158443, 0.38805994, 0.26664197, 0.44283175, 0.085985094, 0.25506780, -0.045542780, 0.33001503, 0.088640086, 0.27517357, 0.11814309, 0.39752114, 0.59512734, 0.60713714, 0.31151596, 0.51146340, 0.070962109, 0.33773333, -0.027466733, 0.26064554, 0.14399286, 0.32974070, 0.080664285, 0.26340839, 0.11261026, 0.67297459, 0.20699884, 0.59673053, 0.18161581, 1.0946180, 0.10080086, 0.56360763, 0.30304310, 0.32767206, 0.17244080, 0.43782458, 0.23742071, 0.38204661, 0.053087011, 0.34085825, 0.33933330, 0.14839207, -0.0033089637, 0.38499305, 0.025951181, 1.2226400, 0.095000707, 0.42016274, 0.21093242, 0.36426690, 0.055057243, 0.20917307, 0.11693041, 0.82800847, 0.028029654, 0.44836789, 0.25682238, 0.76268303, 0.10193096, 0.61659580, 0.013455155, 1.7791119, 0.0067723552, 0.41344157, 0.068857163, 0.42451867, 0.044751827, 0.43863705, 0.14578448, 0.43823594, -0.17383896, 0.61255455, 0.20200206, 0.45367682, 0.087374635, 0.64907575, 0.24012828, 0.62789953, 0.051574931, 0.40460470, 0.0091922851, 0.53881359, -0.0021712238, 0.43065643, 0.22030616, 0.28258345, 0.13933925, 0.33161086, 0.10575007, 0.43598562, -0.020279866, 0.43765736, 0.42020446, 0.21012637, 0.091707572, 0.55183649, 0.26207277, 0.49267027, -0.030590007, 0.35369655, 0.092307486, 0.30298170, 0.083293617, 0.38890216, 0.072379544, 0.23367539, 0.11193615, 0.49026287, 0.47490451, 0.70053118, 0.37591144, 0.51322782, 0.12419472, 0.46957019, 0.016715365, 0.27076674, 0.011771550, 0.76521021, -0.026613414, 0.54392552, 0.070754312, 0.58299339, 0.016248621, 0.49410141, 0.013690999, 2.0493917, -0.15996180, 0.42122555, 0.35677463, 0.71956384, -0.016920041, 0.40862218, 0.17659137, 0.44122782, 0.010538197, 0.48074752, 0.20097928, 0.25842461, -0.015804196, 0.29189819, 0.32103401, 0.32220358, 0.0028490126, 0.47335893, 0.068006516, 0.47349301, 0.011848257, 0.47192028, 0.10796095, 0.28766575, 0.074721560, 0.33987013, 0.13640656, 0.77052331, 0.043711051, 0.54505563, -0.016823160, 1.4419345, -0.10255329, 0.50900733, 0.36979896, 0.41424334, 0.017677745, 0.33573055, 0.028311390, 0.58118784, 0.046248514, 0.47464165, 0.20944417, 0.26943183, -0.031633090, 0.47476298, 0.089825489, 0.98343831, -0.0031536203, 0.37114283, 0.23860134, 0.59679586, 0.066323675, 0.46564266, 0.16391605, 0.69328332, 0.24653515, 0.32620066, 0.21012422, 0.59267676, 0.16176070, 0.39650103, 0.12677082, 1.4275286 };


int judge_rate(int* A_Ac, int len)
{
	bool temp0 = false;
	for (int k = len - 2; k >= 0; k--)
	{
		temp0 |= *(A_Ac + k);
		if (temp0) { break; return 4; }
	}
	if (temp0 == false)
	{
		if (temp0 || *(A_Ac + len - 1)) return 2; //repetition node
		else return 0;  //rate-0 node
	}
	else {
		bool temp1 = true;
		for (int k = 1; k < len; ++k)
		{
			temp1 &= *(A_Ac + k);
			if (!temp1) { break; return 4; }
		}
		if (temp1 == true)
		{
			if (temp1 && *A_Ac) return 1; //rate-1 node
			else return 3; // SPC node
		}
		else return 4;
	}
}
int* PM_sort(float* FilterArray1, int len)
{
	int i, j, cnt = 0;
	int* Posit = new int[len];
	for (i = 0; i < len; i++)
		Posit[i] = 255;
	for (i = 0; i < len; i++)
	{
		cnt = 0;
		for (j = 0; j < len; j++)
			if (i != j)
				if (FilterArray1[i] < FilterArray1[j])
					cnt++;
		//如果此数大于所有的数，则count==14，Posit[14]记录下该数的下标
		while (Posit[cnt] != 255)cnt++;
		//此地已有数据了，则移动到下一个数据区，即count值和前面值一致，说明此数据和占据些地的数据值相等。
		Posit[cnt] = i;
	}
	return Posit;
}
void sort_list_L(int* p, int* q, float* W, int L)
{
	int p_index = 0; int q_index = 0;
	for (int i = 0; i < L; i++) {
		for (int k = 1; k < L; k++)
		{
			if (W[p[p_index]] > W[p[k]]) {
				p_index = k;
			}
			if (W[q[q_index]] < W[q[k]]) {
				q_index = k;
			}
		}
		if (W[p[p_index]] < W[q[q_index]])
		{
			swap(p[p_index], q[q_index]);
			p_index = 0; q_index = 0;

		}
		else { break; }
	}
}
int* list_sub_sort2(float* FilterArray1, int len)
{
	int i, j, cnt = 0;
	int* Posit = new int[len];
	for (i = 0; i < len; i++)
		Posit[i] = 255;
	for (i = 0; i < len; i++)
	{
		cnt = 0;
		for (j = 0; j < len; j++)
			if (i != j)
				if (FilterArray1[i] < FilterArray1[j])
					cnt++;
		//如果此数大于所有的数，则count==14，Posit[14]记录下该数的下标
		while (Posit[cnt] != 255)cnt++;
		//此地已有数据了，则移动到下一个数据区，即count值和前面值一致，说明此数据和占据些地的数据值相等。
		Posit[cnt] = i;
	}
	return Posit;
}
float g_fun(float a, float b, float beta)
{
	float tmp = sgn(a) * sgn(b) * max(min(abs(a), abs(b)) - beta, (float)0);
	return tmp;
}
void decode_BP(float* LLR_y, float** LLR_L, float** LLR_R, int iter, int stage, int* A_Ac, int N, int* u, int* x, int* u_xor, int crc_length, int& iteration)
{
	int block, step;
	int success_flag = 0;
	iteration = 0;

	replace_LLR(LLR_y, LLR_L[stage], N);
	for (int i = 0; i < N; ++i)
	{
		if (A_Ac[i] == 0)
			LLR_R[0][i] = HUGE_VALF;
		else
		{
			LLR_R[0][i] = 0;
		}
	}

	for (int k = 0; k < stage; k++)
	{
		memset(LLR_L[k], 0, N * sizeof(float));
	}
	for (int k = 1; k < stage + 1; k++)
	{
		memset(LLR_R[k], 0, N * sizeof(float));
	}

	for (int it = 0; it < iter; ++it)
	{
		iteration++;
		for (int k = stage - 1; k >= 0; k--)
		{
			block = pow(2, k + 1);
			step = pow(2, k);
			for (int cnt = 0; cnt < N; cnt += block)
			{
				BP_function_OL1(*(LLR_L + k) + cnt, *(LLR_L + k + 1) + cnt, *(LLR_L + k + 1) + cnt + step, *(LLR_R + k) + cnt + step, step);
				BP_function_OL2(*(LLR_L + k) + cnt + step, *(LLR_R + k) + cnt, *(LLR_L + k + 1) + cnt, *(LLR_L + k + 1) + cnt + step, step);
			}
		}
		for (int k = 0; k < stage; k++)
		{
			block = pow(2, k + 1);
			step = pow(2, k);
			for (int cnt = 0; cnt < N; cnt += block)
			{
				BP_function_OR1(*(LLR_R + k + 1) + cnt, *(LLR_R + k) + cnt, *(LLR_L + k + 1) + cnt + step, *(LLR_R + k) + cnt + step, step);
				BP_function_OR2(*(LLR_R + k + 1) + cnt + step, *(LLR_R + k) + cnt, *(LLR_L + k + 1) + cnt, *(LLR_R + k) + cnt + step, step);
			}
		}

		add_hard_SIMD(LLR_L[0], LLR_R[0], u, N);
		add_hard_SIMD(LLR_L[stage], LLR_R[stage], x, N);

		PolarEncode_xor(u_xor, u, N);

		int tmp = 0;

		if (crc_length == 0)
		{
			for (int i = 0; i < N; i++)
			{
				if (u_xor[i] == x[i])
					tmp++;
				else
					break;
			}
			if (tmp == N)
			{
				success_flag = 1;
				break;
			}
		}
		else
		{
			if (it > 0) {
				int tmp = 0;
				for (int i = 0; i < N; i++)
				{
					if (u_xor[i] == x[i])
						tmp++;
					else
						break;
				}
				if (tmp == N)
				{
					success_flag = 1; break;
				}
				else
				{
					success_flag = 0;
				}
			}
		}
	}
}
void decode_BP_oneiter(float* LLR_y, float** LLR_L, float** LLR_R, int iter, int stage, int* A_Ac, int N, int* u, int* x, int* u_xor, int crc_length, int& iteration)
{
	int block, step;
	int success_flag = 0;

	replace_LLR(LLR_y, LLR_L[stage], N);

	for (int it = 0; it < iter; ++it)
	{
		iteration++;
		for (int k = stage - 1; k >= 0; k--)
		{
			block = pow(2, k + 1);
			step = pow(2, k);
			for (int cnt = 0; cnt < N; cnt += block)
			{
				BP_function_OL1(*(LLR_L + k) + cnt, *(LLR_L + k + 1) + cnt, *(LLR_L + k + 1) + cnt + step, *(LLR_R + k) + cnt + step, step);
				BP_function_OL2(*(LLR_L + k) + cnt + step, *(LLR_R + k) + cnt, *(LLR_L + k + 1) + cnt, *(LLR_L + k + 1) + cnt + step, step);
			}
		}
		for (int k = 0; k < stage; k++)
		{
			block = pow(2, k + 1);
			step = pow(2, k);
			for (int cnt = 0; cnt < N; cnt += block)
			{
				BP_function_OR1(*(LLR_R + k + 1) + cnt, *(LLR_R + k) + cnt, *(LLR_L + k + 1) + cnt + step, *(LLR_R + k) + cnt + step, step);
				BP_function_OR2(*(LLR_R + k + 1) + cnt + step, *(LLR_R + k) + cnt, *(LLR_L + k + 1) + cnt, *(LLR_R + k) + cnt + step, step);
			}
		}

		add_hard_SIMD(LLR_L[0], LLR_R[0], u, N);
		add_hard_SIMD(LLR_L[stage], LLR_R[stage], x, N);

	}
}


void r1_tree(int j, int* A_Ac, int N, int K, vector<int>& r1_1)
{
	/*	LLR   N-length LLR as input
	sum   N-length beta value as output
	j     the number of stage, equaling to log2N
	A_Ac  the array which label the channel condition, frozen is 0, information is 1
	a     a variable which label the position the start of each array (alpha and beta)
	N     Codelength
	K     Inoformation bit length  */
	if (j == 0) {     // leaf node
		if (A_Ac[Count] == 1)		r1_1.push_back(Count);
		Count++;
		return;
	}

	int node = 1 << j;
	int length = node >> 1;
	int start = (N << 1) - node;
	int last_start = (N << 1) - (node << 1);
	int type = 4;
	type = judge_rate(A_Ac + Count, node);
	//else type = 4;
	if ((type == 2) || (type == 3)) type = 4;
	switch (type)
	{
	case 0:  //rate-0 node
	{
		Count += node;
		break;
	}
	case 1:  //rate-1 node
	{
		r1_1.push_back(Count);
		Count += node;
		break;
	}
	case 4:  //conventional SC operation
	{
		r1_tree(j - 1, A_Ac, N, K, r1_1);

		r1_tree(j - 1, A_Ac, N, K, r1_1);

		break;
	}
	default:cout << "error" << endl;
	}
	return;
}

void decode(float* LLR, int* sum, int j, int* A_Ac, int a, int N, int K, float* Ma_SC)
{
	/*	LLR   N-length LLR as input
	sum   N-length beta value as output
	j     the number of stage, equaling to log2N
	A_Ac  the array which label the channel condition, frozen is 0, information is 1
	a     a variable which label the position the start of each array (alpha and beta)
	N     Codelength
	K     Inoformation bit length  */
	if (j == 0) {     // leaf node
		if (A_Ac[Count] == 0) {
			*(sum + Count) = 0;
		}
		else {
			*(sum + Count) = hard(*(LLR + (N << 1) - 2));
		}
		Ma_SC[Count] = *(LLR + (N << 1) - 2);
		Count++;
		return;
	}

	int node = 1 << j;
	int length = node >> 1;
	int start = (N << 1) - node;
	int last_start = (N << 1) - (node << 1);
	int type = 4;
	if (j < 6) { //An adaptive operation, if j < 6, use Fast-SSC
		type = judge_rate(A_Ac + Count, node);
	}
	type = 4;
	switch (type)
	{
	case 0:  //rate-0 node
	{
		memset(sum + Count, 0, sizeof(int) * node);
		Count += node;
		break;
	}
	case 1:  //rate-1 node
	{
		hard_SIMD(LLR + last_start, sum + Count, node);
		Count += node;
		break;
	}

	case 2:  //repetition node
	{
		float temp = cal_sum(LLR + last_start, node);
		if (temp >= 0) {
			memset(sum + Count, 0, sizeof(int) * node);
		}
		else {
			for (int i = 0; i < node; i++)
			{
				*(sum + Count + i) = 1;
			}
		}
		Count += node;
		break;
	}
	case 3:  //SPC node
	{
		int parity = 0;
		float temp = abs(*(LLR + last_start));
		int index = 0;
		for (int i = 0; i < node; i++)
		{
			if (*(LLR + last_start + i) >= 0)
			{
				*(sum + Count + i) = 0;
				if (temp > *(LLR + last_start + i)) {
					temp = *(LLR + last_start + i); index = i;
				}
			}
			else
			{
				*(sum + Count + i) = 1;
				parity ^= 1;
				if (temp > abs(*(LLR + last_start + i))) {
					temp = abs(*(LLR + last_start + i)); index = i;
				}
			}
		}
		*(sum + Count + index) ^= parity;
		Count += node;
		break;
	}
	case 4:  //conventional SC operation
	{
		f_function(LLR + last_start, length);
		decode(LLR, sum, j - 1, A_Ac, a, N, K, Ma_SC);
		a = Count >> (j - 1);

		g_function(LLR + last_start, sum + ((a - 1) * length), length);
		decode(LLR, sum, j - 1, A_Ac, a, N, K, Ma_SC);
		a = Count >> (j - 1);
		int initial_posit = (a - 2) << (j - 1);
		if (length < 4) {
			for (int i = 0; i < length; i++)
			{
				*(sum + initial_posit + i) ^= *(sum + initial_posit + length + i);
			}
		}
		else {
			combine(sum + initial_posit, sum + initial_posit + length, length);
		}
		break;
	}
	default:cout << "error" << endl;
	}
	return;
}
int cal_stage(int i, int n)
{
	int stage = 0;
	int N = pow(2, n);
	if (i == 0) stage = n - 1;
	else if (i < N) {
		char* str = new char[n];
		int j = 0;
		do {
			str[j++] = i % 2 + '0';
			i /= 2;
		} while (i);
		for (j = 0; j < n; j++)
		{
			if (str[j] == '1') {
				stage = j; break;
			}
		}
		delete[] str;
	}
	else stage = n - 1;

	return stage;
}
void cal_LLR(int s, float* LLR_new, float* LLR, int* psum, int fg, int N)
{
	int node = 1 << s;
	int length = node >> 1;
	int start = (N << 1) - node;
	int last_start = (N << 1) - (node << 1);
	if (fg == 0) {
		f_function_index(LLR + last_start, LLR_new + last_start, length);
	}
	else {
		g_function_index(LLR + last_start, psum + N - node, LLR_new + last_start, length);
	}
}

void SCAN_decode(float* LLR, float* beta_L, float* beta_R, int* A_Ac, int N, int* st, int* fg)
{
	Count = 0;

	int n = (int)(log(N) / log(2));
	memset(fg, 0, sizeof(int)*(n + 1));

	int node, length, stage, type, start, last_start, beta_start, beta_last_start, betaR_start, betaR_last_start;
	while (Count < N)
	{
		for (int s = st[Count]; s >= 0; s--)
		{
			node = 1 << (s + 1);
			length = node >> 1;

			stage = s;

			start = (N << 1) - node;
			last_start = (N << 1) - (node << 1);

			beta_start = N - node;

			fg[s + 1] ^= 1;
			if (fg[s + 1]) {
				betaR_start = (n - 1 - s)*N / 2 + Count / 2;
				SCAN_function(LLR + start, LLR + last_start, LLR + last_start + length, beta_R + betaR_start, length);
			}
			else {
				betaR_start = (n - 1 - s)*N / 2 + (Count - length) / 2;
				SCAN_function2(LLR + start, LLR + last_start, beta_L + beta_start, LLR + last_start + length, length);
			}
			type = s > 6 ? 4 : judge_rate(A_Ac + Count, length);

			if (s == 0) type = 4;
			//type = 4;
			if (type <= 2) // type <= -1: SCAN// <= 0: R0// <= 1: R0, R1// <= 2: R0, R1, REP// <= 3: Fast
			{
				break;
			}
		}

		switch (type)
		{
		case 0:
		{
			if (fg[stage + 1])
			{
				for (int i = 0; i < length; i++)
					beta_L[beta_start + i] = HUGE_VALF;
			}
			else
			{
				for (int i = 0; i < length; i++)
					beta_R[betaR_start + i] = HUGE_VALF;
			}
			Count += length;
			break;
		}
		case 1:  //rate-1 node
		{
			if (fg[stage + 1])
			{
				memset(beta_L + beta_start, 0, sizeof(float) * length);
			}
			else
			{
				memset(beta_R + betaR_start, 0, sizeof(float) * length);
			}
			Count += length;
			break;
		}
		case 2: //REP node
		{
			float temp = 0;
			if (fg[stage + 1])
			{
				for (int i = 0; i < length; i++)
					temp += LLR[start + i];
				for (int i = 0; i < length; i++)
				{
					beta_L[beta_start + i] = temp - LLR[start + i];
				}
			}
			else
			{
				for (int i = 0; i < length; i++)
					temp += LLR[start + i];
				for (int i = 0; i < length; i++)
				{
					beta_R[betaR_start + i] = temp - LLR[start + i];
				}
			}
			Count += length;
			break;
		}
		default:
		{

			if (A_Ac[Count] == 0) {
				if (fg[1]) beta_L[N - 2] = HUGE_VALF;
				else beta_R[betaR_start] = HUGE_VALF;
			}
			else
			{
				if (fg[1]) beta_L[N - 2] = 0;
				else beta_R[betaR_start] = 0;
			}
			Count++;
			break;
		}
		}

		int t = st[Count];
		if (!fg[stage + 1]) {
			for (int s = stage; s < t; s++)
			{
				node = 1 << (s + 1);
				length = node >> 1;

				start = (N << 1) - node;
				last_start = (N << 1) - (node << 1);

				beta_start = N - node;
				beta_last_start = N - (node << 1);

				betaR_start = (n - 1 - s)*N / 2 + (Count - node) / 2;
				betaR_last_start = (n - 2 - s)*N / 2 + (Count - node * 2) / 2;

				if (s < (t - 1) || ((s == (t - 1)) && (Count == N))) {
					SCAN_function(beta_R + betaR_last_start, beta_L + beta_start, beta_R + betaR_start, LLR + last_start + length, length);
					SCAN_function2(beta_R + betaR_last_start + length, beta_L + beta_start, LLR + last_start, beta_R + betaR_start, length);
				}
				else
				{
					SCAN_function(beta_L + beta_last_start, beta_L + beta_start, beta_R + betaR_start, LLR + last_start + length, length);
					SCAN_function2(beta_L + beta_last_start + length, beta_L + beta_start, LLR + last_start, beta_R + betaR_start, length);
				}
			}
		}
	}
	//BP_function(LLR_R[n], LLR_R[n - 1], LLR_R[n - 1] + N / 2, LLR + N / 2, N / 2);
	//BP_function2(LLR_R[n] + N / 2, LLR_R[n - 1], LLR, LLR_R[n - 1] + N / 2, N / 2);
}
void LDPC_BP_Decoder_SP(float* llr, int** Nv, int* dc, int** Nc, int* dv, int* uhat, int N, int N_Info, int iterations)
{

	double sig = 1;
	double tanhtemp = 0;
	double x = 0;
	double y = 0;
	double** beta = new double* [N - N_Info];
	for (int i = 0; i < N - N_Info; i++)
	{
		beta[i] = new double[N];
	}
	double** alpha = new double* [N];
	for (int i = 0; i < N; i++)
	{
		alpha[i] = new double[N - N_Info];
	}
	double* gamma = new double[N];
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N - N_Info; j++)
		{
			alpha[i][j] = llr[i];
		}
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N - N_Info; j++)
		{
			beta[j][i] = 0;
		}
	}
	for (int epoch = 0; epoch < iterations; epoch++)
	{
		// VN to CN(update CN)
		for (int q = 0; q < N - N_Info; q++)
		{
			for (int i = 0; i < dc[q]; i++)
			{
				sig = 1;
				tanhtemp = 1;//sum-product 
				for (int j = 0; j < dc[q]; j++)
				{
					if (j != i)
					{
						x = alpha[Nc[q][j]][q];
						sig = sig * (2 * (x > 0) - 1);
						tanhtemp = tanhtemp * (tanh(abs(x) / 2)); //sum-product 
					}
				}
				if (tanhtemp < 1e-12)
				{
					beta[q][Nc[q][i]] = 0.0;
				}
				else if (tanhtemp > 1000) {
					beta[q][Nc[q][i]] = sig * 1000.0;//min-sum
				}
				else if (std::isnan(tanhtemp)) {
					beta[q][Nc[q][i]] = 0;
				}
				else {
					beta[q][Nc[q][i]] = 2 * sig * atanh(tanhtemp); //sum-product
				}
			}
		}
		for (int p = 0; p < N; p++)
		{
			gamma[p] = llr[p];
			for (int i = 0; i < dv[p]; i++)
			{
				gamma[p] += beta[Nv[p][i]][p];
			}
			uhat[p] = (gamma[p] < 0);
			for (int i = 0; i < dv[p]; i++)
			{
				alpha[p][Nv[p][i]] = gamma[p] - beta[Nv[p][i]][p];
			}
		}
		/*for (int i = 0; i < 20; i++)
			cout << uhat[i];
		cout << endl;
		for (int i = 0; i < N; i++)
			cout << gamma[i] << ' ';
		cout << endl;
		cout << endl;*/
		int checksum = 0;
		for (int q = 0; q < N - N_Info; q++)
		{
			checksum = 0;
			for (int i = 0; i < dc[q]; i++)
			{
				checksum ^= uhat[Nc[q][i]];
			}
			if (checksum == 1)
				break;
		}
		if (checksum == 0)
		{
			break;
		}
	}
	//exit(0);
	for (int i = 0; i < N - N_Info; i++)
	{
		delete[]beta[i];
	}
	delete[] beta;
	for (int i = 0; i < N; i++)
	{
		delete[]alpha[i];
	}
	delete[] alpha;
	delete[] gamma;
}

void LDPC_BP_Decoder(float* llr,int** Nv, int* dc,int **Nc, int* dv, int* uhat, int N, int N_Info,int iterations)
{

	double sig = 1;
	double tanhtemp = 0;
	double x = 0;
	double y = 0;
	double** beta = new double*[N-N_Info];
	for (int i = 0; i < N - N_Info; i++)
	{
		beta[i] = new double [N];
	}
	double** alpha = new double* [N];
	for (int i = 0; i < N; i++)
	{
		alpha[i] = new double[N - N_Info];
	}
	double* gamma = new double[N];
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N - N_Info; j++)
		{
			alpha[i][j] = llr[i];
		}
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N - N_Info; j++)
		{
			beta[j][i] = 0;
		}
	}
	for (int epoch = 0; epoch < iterations; epoch++)
	{
		// VN to CN(update CN)
		for (int q = 0; q < N - N_Info; q++)
		{
			for (int i = 0; i < dc[q]; i++)
			{
				sig = 1;
				tanhtemp = 100000;//min-sum
				//tanhtemp = 1;//sum-product 
				for (int j = 0; j < dc[q]; j++)
				{
					if (j != i)
					{
						x = alpha[Nc[q][j]][q];
						sig = sig * (2 * (x > 0) - 1);
						tanhtemp = min(abs(tanhtemp), abs(x)); //min-sum
						//tanhtemp = tanhtemp * (tanh(abs(x) / 2)); //sum-product 
					}
				}
				if (tanhtemp == 0)
				{
					beta[q][Nc[q][i]] = 0.0;
				}
				else if (tanhtemp > 1000) {
					beta[q][Nc[q][i]] = sig * 1000.0;//min-sum
				}
				else if (std::isnan(tanhtemp)) {
					beta[q][Nc[q][i]] = 0;
				}
				else {
					beta[q][Nc[q][i]] = sig * tanhtemp; //min-sum
					//beta[q][Nc[q][i]] = 2 * sig * atanh(tanhtemp); //sum-product
				}
			}
		}
		for (int p = 0; p < N; p++)
		{
			gamma[p] = llr[p];
			for (int i = 0; i < dv[p]; i++)
			{
				gamma[p] += beta[Nv[p][i]][p];
			}
			uhat[p] = (gamma[p] < 0);
			for (int i = 0; i < dv[p]; i++)
			{
				alpha[p][Nv[p][i]] = gamma[p] - beta[Nv[p][i]][p];
			}
		}
		//for (int i = 0; i < dc[0]; i++)
			//cout << uhat[Nc[0][i]];
		//cout << endl;
		int checksum = 0;
		for (int q = 0; q < N - N_Info; q++)
		{
			checksum = 0;
			for (int i = 0; i < dc[q]; i++)
			{
				checksum ^= uhat[Nc[q][i]];
			}
			if (checksum == 1)
				break;
		}
		if (checksum == 0) 
		{
			break;
		}
	}
	
	for (int i = 0; i < N - N_Info; i++)
	{
		delete[]beta[i];
	}
	delete[] beta;
	for (int i = 0; i < N; i++)
	{
		delete[]alpha[i];
	}
	delete[] alpha;
	delete[] gamma;
}

void SCAN_decode_hardware(float* LLR_y, float** LLR_L, float** LLR_R, int* A_Ac, int N, float* Ma_SC, float& theta, int& Flip_bit, int* uhat, int* st, int* fg)
{
	Count = 0;

	int n = (int)(log(N) / log(2));
	memset(fg, 0, sizeof(int) * (n + 1));

	replace_LLR(LLR_y, LLR_L[n], N);

	int node, length, stage, type;
	while (Count < N)
	{
		for (int s = st[Count]; s >= 0; s--)
		{
			node = 1 << (s + 1);
			length = node >> 1;

			stage = s;

			fg[s + 1] ^= 1;
			if (fg[s + 1]) {
				//cout << " Count = " << Count << " stage = " << s << " node = " << node << " F function " << endl;

				BP_function(LLR_L[s] + Count, LLR_L[s + 1] + Count, LLR_L[s + 1] + Count + length, LLR_R[s] + Count + length, length);
			}
			else {
				//cout << " Count = " << Count << " stage = " << s << " node = " << node << " G function " << endl;

				BP_function2(LLR_L[s] + Count, LLR_L[s + 1] + Count - length, LLR_R[s] + Count - length, LLR_L[s + 1] + Count, length);
			}
			type = s > 6 ? 4 : judge_rate(A_Ac + Count, length);
			if (s == 0) type = 4;
			//type = 4;
			if (type <= 2) // type <= -1: SCAN// <= 0: R0// <= 1: R0, R1// <= 2: R0, R1, REP// <= 3: Fast
			{
				break;
			}
		}

		switch (type)
		{
		case 0:
		{
			//cout << " R0 node " << " Count = "<<Count<<" stage = " << stage << " node = " << length << endl;
			memset(uhat + Count, 0, sizeof(int) * length);

			for (int i = 0; i < length; i++)
			{
				LLR_R[stage][Count + i] = HUGE_VALF;
			}
			for (int cc = 0; cc < length; cc++)
			{
				Ma_SC[Count + cc] = HUGE_VALF;
			}
			SCAN_R0_FLIP(LLR_L, LLR_R, stage, Count, length, Ma_SC, uhat);

			Count += length;
			break;
		}
		case 1:  //rate-1 node
		{
			SCAN_R1_FLIP(LLR_L, LLR_R, stage, Count, length, Ma_SC, theta, Flip_bit, uhat);

			Count += length;
			break;
		}
		case 2: //REP node
		{
			SCAN_REP_FLIP(LLR_L, LLR_R, stage, Count, length, Ma_SC, theta, Flip_bit, uhat);

			Count += length;
			break;
		}
		case 3:
		{
			SCAN_SPC_FLIP(LLR_L, LLR_R, stage, Count, length, Ma_SC, theta, Flip_bit, uhat);

			Count += length;
			break;
		}
		default:
		{
			if (A_Ac[Count] == 0) {
				uhat[Count] = 0;

				Ma_SC[Count] = HUGE_VALF;
			}
			else
			{
				uhat[Count] = hard(LLR_L[0][Count]);

				Ma_SC[Count] = theta + mthres(LLR_L[0][Count] + LLR_R[0][Count]);//abs(LLR_L[0][Count]);
				//Ma_SC[Count] = abs(LLR_L[0][Count] + LLR_R[0][Count]);
				theta += hthres(LLR_L[0][Count]);

				if (Count == Flip_bit)
				{
					LLR_R[0][Count] = HUGE_VALF * (2 * hard(LLR_L[0][Count]) - 1);
					uhat[Count] ^= 1;
				}
			}
			Count++;
			break;
		}
		}

		int t = st[Count];
		if (!fg[stage + 1]) {
			//if(type <= 0)stage++;
			for (int s = stage; s < t; s++)
			{
				node = 1 << (s + 1);
				length = node >> 1;

				if (length < 8)
				{
					for (int i = 0; i < length; ++i)
					{
						//LLR_R[s + 1][Count - node + i] = sgn(LLR_R[s][Count - node + i]) * sgn(LLR_R[s][Count - length + i] + LLR_L[s + 1][Count - length + i]) * min(abs(LLR_R[s][Count - node + i]), abs(LLR_R[s][Count - length + i] + LLR_L[s + 1][Count - length + i]));
						//LLR_R[s + 1][Count - length + i] = sgn(LLR_R[s][Count - node + i]) * sgn(LLR_L[s + 1][Count - node + i]) * min(abs(LLR_R[s][Count - node + i]), abs(LLR_L[s + 1][Count - node + i])) + LLR_R[s][Count - length + i];
						LLR_R[s + 1][Count - node + i] = sgn(LLR_R[s][Count - node + i]) * sgn(LLR_R[s][Count - length + i] + LLR_L[s + 1][Count - length + i]) * max(min(abs(LLR_R[s][Count - node + i]), abs(LLR_R[s][Count - length + i] + LLR_L[s + 1][Count - length + i])) - 0.25, 0.0);
						LLR_R[s + 1][Count - length + i] = sgn(LLR_R[s][Count - node + i]) * sgn(LLR_L[s + 1][Count - node + i]) * max(min(abs(LLR_R[s][Count - node + i]), abs(LLR_L[s + 1][Count - node + i])) - 0.25, 0.0) + LLR_R[s][Count - length + i];

					}
				}
				//BP_function(LLR_R[s + 1] + Count - node, LLR_R[s] + Count - node, LLR_R[s] + Count - length, LLR_L[s + 1] + Count - length, length);
				//BP_function2(LLR_R[s + 1] + Count - length, LLR_R[s] + Count - node, LLR_L[s + 1] + Count - node, LLR_R[s] + Count - length, length);
				BP_function_OR1(LLR_R[s + 1] + Count - node, LLR_R[s] + Count - node, LLR_R[s] + Count - length, LLR_L[s + 1] + Count - length, length);
				BP_function_OR2(LLR_R[s + 1] + Count - length, LLR_R[s] + Count - node, LLR_L[s + 1] + Count - node, LLR_R[s] + Count - length, length);
			}
		}
	}
	BP_function(LLR_R[n], LLR_R[n - 1], LLR_R[n - 1] + N / 2, LLR_L[n] + N / 2, N / 2);
	BP_function2(LLR_R[n] + N / 2, LLR_R[n - 1], LLR_L[n], LLR_R[n - 1] + N / 2, N / 2);
}
void decode_list_with_no_copy(float** LLR, int** psum_L, int** psum_R, int j, int* A_Ac, int a, int N, int K, float* PM, int L, int m, int** p, int* fg,
	float* LLR_in, float* W, int* index, int* better, int* worse, int* path_number, int* st)
{
	Count = 0;
	Count_info = 0;
	l = 1;
	int n = (int)(log(N) / log(2));
	memset(fg, 0, sizeof(int) * (n + 1));

	//for (Count = 0; Count < N; Count++)
	int node, length, start, last_start;
	int type;
	int stage;
	while (Count < N)
	{
		for (int s = st[Count]; s >= 0; s--)
		{
			node = 1 << (s + 1);
			length = node >> 1;
			start = (N << 1) - node;
			last_start = (N << 1) - (node << 1);

			stage = s;

			fg[s + 1] ^= 1;
			if (fg[s + 1]) {
				for (int k = 0; k < l; k++)
				{
					f_function_index(*(LLR + p[k][s + 1]) + last_start, *(LLR + k) + last_start, length);
					p[k][s] = k;
				}
			}
			else {
				for (int k = 0; k < l; k++) {
					g_function_index(*(LLR + p[k][s + 1]) + last_start, *(psum_L + k) + N - node, *(LLR + k) + last_start, length);
					//if(k!=p[k][s+1])cout << " Count - " << Count << " k = " << k << " pk = " << p[k][s+1] << endl;
					p[k][s] = k;
				}
			}
			type = s > 8 ? 4 : judge_rate(A_Ac + Count, length);
			if ((Count_info < m) && ((type == 1)))
			{
				type = 4;
			}
			if (s == 0) type = 4;
			if (type <= 2)
			{
				break;
			}
		}
		switch (type)
		{
		case 0:
		{
			if (fg[stage + 1])
			{
				R0_list(LLR, psum_L, length, l, start - N, start, PM, p, stage, Count_info);
			}
			else
			{
				R0_list(LLR, psum_R, length, l, start - N, start, PM, p, stage, Count_info);
			}
			Count += length;
			break;
		}
		case 1:
		{
			if (fg[stage + 1])
			{
				R1_list(LLR, psum_L, length, L, start - N, start, PM, n, L, p, W, index, better, worse, stage);
			}
			else
			{
				R1_list(LLR, psum_R, length, L, start - N, start, PM, n, L, p, W, index, better, worse, stage);
			}
			Count += length;
			Count_info += length;
			break;
		}
		case 2:
		{
			if (fg[stage + 1])
			{
				REP_list(LLR, psum_L, length, l, start - N, start, PM, n, L, p, W, index, better, worse, stage);
			}
			else
			{
				REP_list(LLR, psum_R, length, l, start - N, start, PM, n, L, p, W, index, better, worse, stage);
			}
			Count += length;
			Count_info++;
			break;
		}
		default:
			if (A_Ac[Count] == 0)  //frozen bit condition
			{
				for (int k = 0; k < l; k++) {
					if (Count_info > 0) {
						if (LLR[k][2 * N - 2] < 0) { PM[k] = PM[k] + LLR[k][2 * N - 2]; }
					}
					if (fg[1]) psum_L[k][N - 2] = 0;
					else psum_R[k][N - 2] = 0;
				}
				Count++;
			}

			else {
				Count_info++;
				if (Count_info <= m)
				{
					for (int k = l - 1; k >= 0; k--) {
						LLR_in[k] = *(*(LLR + k) + (N << 1) - 2);
						if (LLR_in[k] >= 0) {
							PM[(k << 1) + 1] = PM[k] - LLR_in[k];
							PM[(k << 1)] = PM[k];
						}
						else {
							PM[(k << 1) + 1] = PM[k];
							PM[(k << 1)] = PM[k] + LLR_in[k];
						}
						if (fg[1]) {
							psum_L[(k << 1) + 1][N - 2] = 1;
							psum_L[(k << 1)][N - 2] = 0;
						}
						else {
							psum_R[(k << 1) + 1][N - 2] = 1;
							psum_R[(k << 1)][N - 2] = 0;
						}
						for (int s = 0; s <= n; s++) {
							p[(k << 1) + 1][s] = p[k][s];
							p[(k << 1)][s] = p[k][s];
						}
					}
					l <<= 1;
					if (l > L)l = L;
				}
				else {
					for (int k = 0; k < L; k++)
					{
						LLR_in[k] = *(*(LLR + k) + (N << 1) - 2);

						if (LLR_in[k] >= 0) {
							W[(k << 1)] = PM[k];
							W[(k << 1) + 1] = PM[k] - LLR_in[k];
							better[k] = (k << 1);
							worse[k] = (k << 1) + 1;
						}
						else {
							W[(k << 1)] = PM[k] + LLR_in[k];
							W[(k << 1) + 1] = PM[k];
							worse[k] = (k << 1);
							better[k] = (k << 1) + 1;
						}
					}
					if (Count < N - 1) {
						sort_list_L(better, worse, W, L);  // Path metric sortig
						for (int k = 0; k < L; k++) {
							index[k] = (better[k] >> 1);
							if (index[k] != k) {
								for (int s = 0; s <= n; s++) {
									p[k][s] = p[index[k]][s];
								}
							}
						}
						for (int k = 0; k < L; k++) {  //path updating
							if (fg[1])psum_L[k][N - 2] = better[k] % 2;
							else psum_R[k][N - 2] = better[k] % 2;
							PM[k] = W[better[k]];
						}
					}
					else {
						int* Posit = list_sub_sort2(W, 2 * L);
						for (int k = 0; k < L; k++) {
							PM[k] = W[Posit[k]];
							index[k] = (int)Posit[k] / 2;
						}
						for (int s = 0; s <= n; s++) {
							for (int k = 0; k < L; k++) {
								worse[k] = p[index[k]][s];
							}
							for (int k = 0; k < L; k++) {
								p[k][s] = worse[k];
							}
						}
						for (int k = 0; k < L; k++) {
							psum_R[k][N - 2] = Posit[k] % 2;
						}

					}
				}
				Count++;
			}
			break;
		}

		int t = st[Count];
		if (!fg[stage + 1]) {
			//if(type <= 0)stage++;
			for (int s = stage; s < t; s++)
			{
				node = 1 << (s + 1);
				length = node >> 1;
				start = N - node;
				last_start = N - (node << 1);

				if (s < (t - 1) || ((s == (t - 1)) && (Count == N))) {
					for (int k = 0; k < l; k++)
					{
						if (length < 4) {
							for (int i = 0; i < length; i++)
							{
								*(*(psum_R + k) + last_start + i) = *(*(psum_L + p[k][s]) + start + i) ^ *(*(psum_R + k) + start + i);
								*(*(psum_R + k) + last_start + length + i) = *(*(psum_R + k) + start + i);
							}
						}
						else {
							combine_index(*(psum_L + p[k][s]) + start, *(psum_R + k) + start, *(psum_R + k) + last_start, length);
						}
						p[k][s] = k;
					}
				}
				else {
					for (int k = 0; k < l; k++)
					{
						if (length < 4) {
							for (int i = 0; i < length; i++)
							{
								*(*(psum_L + k) + last_start + i) = *(*(psum_L + p[k][s]) + start + i) ^ *(*(psum_R + k) + start + i);
								*(*(psum_L + k) + last_start + length + i) = *(*(psum_R + k) + start + i);
							}
						}
						else {
							combine_index(*(psum_L + p[k][s]) + start, *(psum_R + k) + start, *(psum_L + k) + last_start, length);
						}
						p[k][s] = k;
					}
				}
			}
		}
	}
}

void SCAN_decode_hardware_list(float** LLR, float** beta_L, float** beta_R, int* A_Ac, int N,
	int* st, int* fg, float* PM, int L, int m, int** p,
	float* LLR_in, float* W, int* index, int* better, int* worse, int* path_number)
{
	Count = 0;
	Count_info = 0;
	l = 1;

	int n = (int)(log(N) / log(2));
	memset(fg, 0, sizeof(int)*(n + 1));

	int node, length, stage, type, start, last_start, beta_start, beta_last_start, betaR_start, betaR_last_start;
	while (Count < N)
	{
		for (int s = st[Count]; s >= 0; s--)
		{
			node = 1 << (s + 1);
			length = node >> 1;

			stage = s;

			start = (N << 1) - node;
			last_start = (N << 1) - (node << 1);

			beta_start = N - node;

			fg[s + 1] ^= 1;
			if (fg[s + 1]) {
				betaR_start = (n - 1 - s)*N / 2 + Count / 2;
				for (int k = 0; k < l; k++)
				{
					SCAN_function(*(LLR + k) + start, *(LLR + p[k][s + 1]) + last_start, *(LLR + p[k][s + 1]) + last_start + length, *(beta_R + k) + betaR_start, length);
					p[k][s] = k;
				}
			}
			else {
				betaR_start = (n - 1 - s)*N / 2 + (Count - length) / 2;
				for (int k = 0; k < l; k++)
				{
					SCAN_function2(*(LLR + k) + start, *(LLR + p[k][s + 1]) + last_start, *(beta_L + k) + beta_start, *(LLR + p[k][s + 1]) + last_start + length, length);
					p[k][s] = k;
				}
			}
			type = s > 6 ? 4 : judge_rate(A_Ac + Count, length);

			if (s == 0) type = 4;
			if ((Count_info < m) && ((type == 1)))
			{
				type = 4;
			}
			//if ((type == 1))type = 4;

			if (type <= 2) // type <= -1: SCAN// <= 0: R0// <= 1: R0, R1// <= 2: R0, R1, REP// <= 3: Fast
			{
				break;
			}
		}

		switch (type)
		{
		case 0:
		{
			if (fg[stage + 1])
			{
				SCAN_R0_LIST(LLR, beta_L, beta_start, Count, length, l, start, PM, Count_info);
			}
			else
			{
				SCAN_R0_LIST(LLR, beta_R, betaR_start, Count, length, l, start, PM, Count_info);
			}
			Count += length;
			break;
		}
		case 1:
		{
			if (L <= 2)
			{
				if (fg[stage + 1])
				{
					SCAN_R1_LIST(LLR, beta_L, length, l, beta_start, start, PM, n, L, p, W, index, better, worse, path_number, stage);
				}
				else
				{
					SCAN_R1_LIST(LLR, beta_R, length, l, betaR_start, start, PM, n, L, p, W, index, better, worse, path_number, stage);
				}
			}
			else
			{
				if (fg[stage + 1])
				{
					SCAN_R1_LIST_2(LLR, beta_L, length, l, beta_start, start, PM, n, L, p, W, index, better, worse, path_number, stage);
				}
				else
				{
					SCAN_R1_LIST_2(LLR, beta_R, length, l, betaR_start, start, PM, n, L, p, W, index, better, worse, path_number, stage);
				}
			}
			Count += length;
			Count_info += length;
			break;
		}
		case 2:
		{
			if (fg[stage + 1])
			{
				SCAN_REP_LIST(LLR, beta_L, length, l, beta_start, start, PM, n, L, p, W, index, better, worse, stage);
			}
			else
			{
				SCAN_REP_LIST(LLR, beta_R, length, l, betaR_start, start, PM, n, L, p, W, index, better, worse, stage);
			}
			Count += length;
			Count_info++;
			break;
		}
		default:
		{
			if (A_Ac[Count] == 0) {
				for (int k = 0; k < l; k++) {
					if (Count_info > 0) {
						if (LLR[k][2 * N - 2] < 0) { PM[k] = PM[k] + LLR[k][2 * N - 2]; }
					}
					if (fg[1]) beta_L[k][N - 2] = HUGE_VALF;
					else beta_R[k][betaR_start] = HUGE_VALF;
				}
				Count++;
			}
			else
			{
				Count_info++;
				if (Count_info <= m)
				{
					for (int k = l - 1; k >= 0; k--) {
						LLR_in[k] = *(*(LLR + k) + (N << 1) - 2);
						if (LLR_in[k] >= 0) {
							PM[(k << 1) + 1] = PM[k] - LLR_in[k];
							PM[(k << 1)] = PM[k];
						}
						else {
							PM[(k << 1) + 1] = PM[k];
							PM[(k << 1)] = PM[k] + LLR_in[k];
						}
						if (fg[1]) {
							beta_L[(k << 1) + 1][N - 2] = -HUGE_VALF;
							beta_L[(k << 1)][N - 2] = HUGE_VALF;
						}
						else {
							beta_R[(k << 1) + 1][betaR_start] = -HUGE_VALF;
							beta_R[(k << 1)][betaR_start] = HUGE_VALF;
						}
						for (int s = 0; s <= n; s++) {
							p[(k << 1) + 1][s] = p[k][s];
							p[(k << 1)][s] = p[k][s];
						}
					}
					l <<= 1;
					if (l > L)l = L;
				}
				else
				{
					for (int k = 0; k < L; k++)
					{
						LLR_in[k] = *(*(LLR + k) + (N << 1) - 2);

						if (LLR_in[k] >= 0) {
							W[(k << 1)] = PM[k];
							W[(k << 1) + 1] = PM[k] - LLR_in[k];
							better[k] = (k << 1);
							worse[k] = (k << 1) + 1;
						}
						else {
							W[(k << 1)] = PM[k] + LLR_in[k];
							W[(k << 1) + 1] = PM[k];
							worse[k] = (k << 1);
							better[k] = (k << 1) + 1;
						}
					}
					//if (Count < N - 1) {
					if(R1F[Count] == 1){
						sort_list_L(better, worse, W, L);  // Path metric sortig
						for (int k = 0; k < L; k++) {
							index[k] = (better[k] >> 1);
							if (index[k] != k) {
								for (int s = 0; s <= n; s++) {
									p[k][s] = p[index[k]][s];
								}
							}
						}
						for (int k = 0; k < L; k++) {  //path updating
							if (fg[1])beta_L[k][N - 2] = (1 - 2 * (better[k] % 2))*HUGE_VALF;
							else beta_R[k][betaR_start] = (1 - 2 * (better[k] % 2))*HUGE_VALF;
							PM[k] = W[better[k]];
						}
					}
					else
					{
						for (int k = 0; k < L; k++) {  //path updating
							if (fg[1])beta_L[k][N - 2] = (1 - 2 * (better[k] % 2))*0;
							else beta_R[k][betaR_start] = (1 - 2 * (better[k] % 2))*0;
						}
					}
					/*else {
						int * Posit = list_sub_sort2(W, 2 * L);
						for (int k = 0; k < L; k++) {
							PM[k] = W[Posit[k]];
							index[k] = (int)Posit[k] / 2;
						}
						for (int s = 0; s <= n; s++) {
							for (int k = 0; k < L; k++) {
								worse[k] = p[index[k]][s];
							}
							for (int k = 0; k < L; k++) {
								p[k][s] = worse[k];
							}
						}
						for (int k = 0; k < L; k++) {
							beta_R[k][betaR_start] = (1 - 2 * (Posit[k] % 2))*HUGE_VALF;
						}
					}*/
				}
				Count++;
			}
			break;
		}
		}

		int t = st[Count];
		if (!fg[stage + 1]) {
			for (int s = stage; s < t; s++)
			{
				node = 1 << (s + 1);
				length = node >> 1;

				start = (N << 1) - node;
				last_start = (N << 1) - (node << 1);

				beta_start = N - node;
				beta_last_start = N - (node << 1);

				betaR_start = (n - 1 - s)*N / 2 + (Count - node) / 2;
				betaR_last_start = (n - 2 - s)*N / 2 + (Count - node * 2) / 2;

				if (s < (t - 1) || ((s == (t - 1)) && (Count == N))) {
					for (int k = 0; k < l; k++)
					{
						SCAN_function(*(beta_R + k) + betaR_last_start, *(beta_L + p[k][s]) + beta_start, *(beta_R + k) + betaR_start, *(LLR + p[k][s + 1]) + last_start + length, length);
						SCAN_function2(*(beta_R + k) + betaR_last_start + length, *(beta_L + p[k][s]) + beta_start, *(LLR + p[k][s + 1]) + last_start, *(beta_R + k) + betaR_start, length);
						p[k][s] = k;
					}
				}
				else
				{
					for (int k = 0; k < l; k++)
					{
						SCAN_function(*(beta_L + k) + beta_last_start, *(beta_L + p[k][s]) + beta_start, *(beta_R + k) + betaR_start, *(LLR + p[k][s + 1]) + last_start + length, length);
						SCAN_function2(*(beta_L + k) + beta_last_start + length, *(beta_L + p[k][s]) + beta_start, *(LLR + p[k][s + 1]) + last_start, *(beta_R + k) + betaR_start, length);
						p[k][s] = k;
					}
				}
			}
		}
	}
}
